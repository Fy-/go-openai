package openai

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"errors"

	utils "github.com/sashabaranov/go-openai/internal"
)

var (
	headerData  = regexp.MustCompile(`^data:\s*`)
	errorPrefix = regexp.MustCompile(`^data:\s*{"error":`)
)

type streamable interface {
	ChatCompletionStreamResponse | CompletionResponse
}

type streamReader[T streamable] struct {
	emptyMessagesLimit uint
	isFinished         bool
	receivedDone       bool // Track if we received the [DONE] marker

	reader         *bufio.Reader
	response       *http.Response
	errAccumulator utils.ErrorAccumulator
	unmarshaler    utils.Unmarshaler

	httpHeader
}

func (stream *streamReader[T]) Recv() (response T, err error) {
	rawLine, err := stream.RecvRaw()
	if err != nil {
		// Check for common network errors that might cause unexpected EOF
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			// Return EOF consistently for all EOF-like errors
			return response, io.EOF
		}
		return
	}

	err = stream.unmarshaler.Unmarshal(rawLine, &response)
	if err != nil {
		return
	}
	return response, nil
}

func (stream *streamReader[T]) RecvRaw() ([]byte, error) {
	if stream.isFinished {
		return nil, io.EOF
	}

	return stream.processLines()
}

//nolint:gocognit
func (stream *streamReader[T]) processLines() ([]byte, error) {
	var (
		emptyMessagesCount uint
		hasErrorPrefix     bool
	)

	for {
		rawLine, readErr := stream.reader.ReadBytes('\n')
		
		// Handle EOF with partial data
		if readErr == io.EOF && len(rawLine) > 0 {
			// Process the partial line first
			noSpaceLine := bytes.TrimSpace(rawLine)
			if headerData.Match(noSpaceLine) {
				noPrefixLine := headerData.ReplaceAll(noSpaceLine, nil)
				if string(noPrefixLine) != "[DONE]" {
					// Return the partial data
					return noPrefixLine, nil
				}
			}
			// Check if we have any accumulated error data that might indicate incomplete stream
			if len(stream.errAccumulator.Bytes()) > 0 {
				// We have partial data in error accumulator, might be incomplete JSON
				return nil, fmt.Errorf("stream ended unexpectedly with partial data")
			}
			// After processing partial data, return EOF
			stream.isFinished = true
			return nil, io.EOF
		}
		
		if readErr != nil || hasErrorPrefix {
			// Check if it's a real EOF (end of stream marker)
			if readErr == io.EOF && !hasErrorPrefix && len(stream.errAccumulator.Bytes()) == 0 {
				stream.isFinished = true
				return nil, io.EOF
			}
			
			respErr := stream.unmarshalError()
			if respErr != nil {
				return nil, fmt.Errorf("error, %w", respErr.Error)
			}
			return nil, readErr
		}

		noSpaceLine := bytes.TrimSpace(rawLine)
		if errorPrefix.Match(noSpaceLine) {
			hasErrorPrefix = true
		}
		if !headerData.Match(noSpaceLine) || hasErrorPrefix {
			if hasErrorPrefix {
				noSpaceLine = headerData.ReplaceAll(noSpaceLine, nil)
			}
			writeErr := stream.errAccumulator.Write(noSpaceLine)
			if writeErr != nil {
				return nil, writeErr
			}
			emptyMessagesCount++
			if emptyMessagesCount > stream.emptyMessagesLimit {
				return nil, ErrTooManyEmptyStreamMessages
			}

			continue
		}

		noPrefixLine := headerData.ReplaceAll(noSpaceLine, nil)
		if string(noPrefixLine) == "[DONE]" {
			stream.isFinished = true
			stream.receivedDone = true
			return nil, io.EOF
		}

		return noPrefixLine, nil
	}
}

func (stream *streamReader[T]) unmarshalError() (errResp *ErrorResponse) {
	errBytes := stream.errAccumulator.Bytes()
	if len(errBytes) == 0 {
		return
	}

	err := stream.unmarshaler.Unmarshal(errBytes, &errResp)
	if err != nil {
		errResp = nil
	}

	return
}

func (stream *streamReader[T]) Close() error {
	return stream.response.Body.Close()
}

// IsComplete returns true if the stream received the [DONE] marker
func (stream *streamReader[T]) IsComplete() bool {
	return stream.receivedDone
}
