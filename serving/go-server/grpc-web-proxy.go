package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	pb "ssm-inference-server/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ProxyServer struct {
	grpcClient pb.InferenceServiceClient
}

func NewProxyServer() (*ProxyServer, error) {
	// Get the gRPC server address from environment variable or use default
	grpcServer := "inference-server:50051"
	if envServer := os.Getenv("GRPC_SERVER"); envServer != "" {
		grpcServer = envServer
	}
	
	conn, err := grpc.Dial(grpcServer, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	
	return &ProxyServer{
		grpcClient: pb.NewInferenceServiceClient(conn),
	}, nil
}

func (p *ProxyServer) handleInference(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		AudioData  []byte `json:"audioData"`
		TextPrompt string `json:"textPrompt"`
		MaxTokens  int32  `json:"maxTokens"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Convert to gRPC request
	grpcReq := &pb.GenerateRequest{
		AudioData:  req.AudioData,
		TextPrompt: req.TextPrompt,
		MaxTokens:  req.MaxTokens,
	}

	// Call gRPC service
	resp, err := p.grpcClient.Generate(context.Background(), grpcReq)
	if err != nil {
		log.Printf("gRPC call failed: %v", err)
		http.Error(w, "Inference failed", http.StatusInternalServerError)
		return
	}

	// Convert response back to JSON
	jsonResp := map[string]interface{}{
		"generatedText":   resp.GeneratedText,
		"confidenceScore": resp.ConfidenceScore,
		"latencyMs":       resp.LatencyMs,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jsonResp)
}

func (p *ProxyServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp, err := p.grpcClient.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		http.Error(w, "Health check failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": resp.Status})
}

func main() {
	proxy, err := NewProxyServer()
	if err != nil {
		log.Fatalf("Failed to create proxy: %v", err)
	}

	http.HandleFunc("/api/inference", proxy.handleInference)
	http.HandleFunc("/api/health", proxy.handleHealth)

	log.Println("gRPC-Web proxy listening on :8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
} 