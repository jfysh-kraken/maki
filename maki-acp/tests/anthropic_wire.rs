//! Recorded Anthropic wire bytes through HTTP, provider decoding, agent and ACP.
mod support;

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn anthropic_refusal_wire_reaches_acp() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let mut request = vec![0; 65536];
                let _ = socket.read(&mut request).await.unwrap();
                let body = concat!(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
                    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            });
        }
    });
    let mut child = support::MakiChild::spawn_with_env(&[
        ("ANTHROPIC_API_KEY", "fixture-no-network"),
        ("ANTHROPIC_BASE_URL", &base),
    ])
    .await
    .unwrap();
    child.initialize().await.unwrap();
    let project = tempfile::TempDir::new().unwrap();
    let cwd = project.path().to_str().unwrap();
    let allocated = child.allocate_raw(cwd).await.unwrap();
    let sid = allocated["result"]["sessionId"].as_str().unwrap();
    let activated = child
        .activate_raw(
            sid,
            "anthropic/claude-sonnet-4-6",
            "build",
            cwd,
            project.path().join("result.md").to_str().unwrap(),
        )
        .await
        .unwrap();
    assert!(activated.get("error").is_none(), "{activated}");
    let id = child.fresh_id();
    child
        .send_request(
            "session/prompt",
            id.clone(),
            json!({"sessionId":sid,"prompt":[{"type":"text","text":"fixture"}]}),
        )
        .await
        .unwrap();
    let reply = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        child.expect_response(&id),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(reply["result"]["stopReason"], "refusal", "{reply}");
    assert_eq!(
        reply["result"]["_meta"]["maki.rawReason"], "refusal",
        "{reply}"
    );
    child.shutdown().await.unwrap();
    server.abort();
}
