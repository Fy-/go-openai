package openai

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"net/http"
	"regexp"

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
	dataBuffer     *bytes.Buffer // Buffer for accumulating multi-line data

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

	// Check if the response is an error response
	if bytes.Contains(rawLine, []byte(`"error":`)) {
		var errResp ErrorResponse
		if err = stream.unmarshaler.Unmarshal(rawLine, &errResp); err == nil && errResp.Error != nil {
			return response, errResp.Error
		}
	}

	// Try to unmarshal the response
	err = stream.unmarshaler.Unmarshal(rawLine, &response)
	if err != nil {
		// SGLang might send partial JSON for structured output streaming
		// Log but don't fail if it's a parsing error
		if bytes.Contains(rawLine, []byte(`"choices"`)) {
			// Likely a valid response with parsing issues, return empty response
			return response, nil
		}
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
	// Initialize data buffer if needed
	if stream.dataBuffer == nil {
		stream.dataBuffer = new(bytes.Buffer)
	}

	var emptyMessagesCount uint

	for {
		rawLine, readErr := stream.reader.ReadBytes('\n')
		
		// Handle read errors
		if readErr != nil {
			if readErr == io.EOF {
				// Check if we have accumulated data
				if stream.dataBuffer.Len() > 0 {
					data := stream.dataBuffer.Bytes()
					stream.dataBuffer.Reset()
					return data, nil
				}
				stream.isFinished = true
				return nil, io.EOF
			}
			return nil, readErr
		}

		line := bytes.TrimRight(rawLine, "\r\n")
		
		// Empty line signals end of an event
		if len(line) == 0 {
			// Check if we have accumulated error data
			if stream.errAccumulator.Bytes() != nil && len(stream.errAccumulator.Bytes()) > 0 {
				respErr := stream.unmarshalError()
				if respErr != nil {
					return nil, respErr.Error
				}
			}
			
			if stream.dataBuffer.Len() > 0 {
				// We have a complete event
				data := stream.dataBuffer.Bytes()
				stream.dataBuffer.Reset()
				// SGLang sometimes sends incomplete JSON chunks for structured output
				// Skip validation here, let the unmarshaler handle it
				return data, nil
			}
			emptyMessagesCount++
			if emptyMessagesCount > stream.emptyMessagesLimit {
				return nil, ErrTooManyEmptyStreamMessages
			}
			continue
		}

		// Check for data: prefix
		if bytes.HasPrefix(line, []byte("data: ")) {
			data := bytes.TrimPrefix(line, []byte("data: "))
			
			// Check for [DONE] marker or SGLang's done indicator
			dataStr := string(data)
			if dataStr == "[DONE]" || dataStr == "done" {
				stream.isFinished = true
				stream.receivedDone = true
				return nil, io.EOF
			}
			
			// SGLang might send empty data as heartbeat
			if dataStr == "" {
				// Continue accumulating, might be a heartbeat
				continue
			}
			
			// Accumulate data (handles multi-line data)
			if stream.dataBuffer.Len() > 0 {
				stream.dataBuffer.WriteByte('\n') // Add newline between data lines
			}
			stream.dataBuffer.Write(data)
		} else if bytes.HasPrefix(line, []byte("error: ")) {
			// Handle error events
			errData := bytes.TrimPrefix(line, []byte("error: "))
			stream.errAccumulator.Write(errData)
		} else if bytes.Contains(line, []byte(`"error":`)) && bytes.HasPrefix(line, []byte("{")) {
			// Handle raw JSON error (backward compatibility)
			stream.errAccumulator.Write(line)
			stream.errAccumulator.Write([]byte("\n"))
		}
		// Ignore other event types (like event:, id:, retry:)
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
