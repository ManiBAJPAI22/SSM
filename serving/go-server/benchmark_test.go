// serving/go-server/benchmark_test.go
package main

import (
	"context"
	"testing"
	"time"

	pb "ssm-inference-server/pb"
)

func BenchmarkInference(b *testing.B) {
    server := &InferenceServer{/* initialize */}
    
    req := &pb.GenerateRequest{
        AudioData:  generateDummyAudio(),
        TextPrompt: "Test prompt",
        MaxTokens:  50,
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := server.Generate(context.Background(), req)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func generateDummyAudio() []byte {
    // Generate dummy audio data for testing
    audioData := make([]byte, 16000) // 1 second of audio at 16kHz
    for i := range audioData {
        audioData[i] = byte(i % 256)
    }
    return audioData
}

func TestLatencyTarget(t *testing.T) {
    // Target: < 1000ms per request
    server := &InferenceServer{/* initialize */}
    
    start := time.Now()
    _, err := server.Generate(context.Background(), &pb.GenerateRequest{
        AudioData:  generateDummyAudio(),
        TextPrompt: "Latency test",
        MaxTokens:  10,
    })
    
    if err != nil {
        t.Fatal(err)
    }
    
    elapsed := time.Since(start)
    if elapsed > time.Millisecond*1000 {
        t.Errorf("Latency target missed: %v > 1000ms", elapsed)
    }
}
