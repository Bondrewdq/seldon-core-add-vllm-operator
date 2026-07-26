package main

import "encoding/json"

type SeldonRequest struct {
	Data     *SeldonData            `json:"data,omitempty"`
	StrData  string                 `json:"strData,omitempty"`
	JsonData json.RawMessage        `json:"jsonData,omitempty"`
	Meta     map[string]interface{} `json:"meta,omitempty"`
}

type SeldonData struct {
	Names   []string        `json:"names,omitempty"`
	Ndarray json.RawMessage `json:"ndarray,omitempty"`
}

type SeldonResponse struct {
	JsonData map[string]interface{} `json:"jsonData,omitempty"`
	Meta     map[string]interface{} `json:"meta,omitempty"`
	Status   *SeldonStatus          `json:"status,omitempty"`
}

type SeldonStatus struct {
	Code   int    `json:"code"`
	Info   string `json:"info"`
	Status string `json:"status"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type openAIChoice struct {
	Index        int          `json:"index,omitempty"`
	Message      *ChatMessage `json:"message,omitempty"`
	Text         string       `json:"text,omitempty"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

type openAIResponse struct {
	ID      string         `json:"id,omitempty"`
	Object  string         `json:"object,omitempty"`
	Created int64          `json:"created,omitempty"`
	Model   string         `json:"model,omitempty"`
	Choices []openAIChoice `json:"choices,omitempty"`
	Usage   *Usage         `json:"usage,omitempty"`
}

type adapterError struct {
	statusCode int
	reason     string
	message    string
}

func (e *adapterError) Error() string {
	return e.message
}
