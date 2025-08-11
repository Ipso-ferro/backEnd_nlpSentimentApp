# app.py
# pip install flask
from flask import Flask, request, jsonify, Response
from openai_trained import classify  # <-- make sure this path/module is correct

app = Flask(__name__)

@app.get("/")
def index():
    # HTML page with Bootstrap + pretty rendering
    html = """
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8"/>
  <title>Sentiment Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <!-- Bootstrap CSS (CDN) -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="py-5">
  <div class="container" style="max-width: 760px;">
    <h1 class="mb-4">Sentiment Classifier</h1>
    <div class="card shadow-sm">
      <div class="card-body">
        <form id="form" class="mb-3">
          <div class="mb-3">
            <label for="text" class="form-label">Enter a comment</label>
            <textarea id="text" class="form-control" rows="4"
              placeholder="e.g. 'awful support, totally broken and late'"></textarea>
          </div>
          <button id="btn" type="submit" class="btn btn-primary">
            <span class="spinner-border spinner-border-sm me-2 d-none" id="spin"></span>
            Classify
          </button>
        </form>
        <div id="out" class="alert d-none" role="alert"></div>
        <pre id="json" class="bg-light p-3 rounded d-none"></pre>
      </div>
    </div>
  </div>

  <!-- Bootstrap JS (CDN, includes Popper) -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <script>
    const form  = document.getElementById('form');
    const btn   = document.getElementById('btn');
    const ta    = document.getElementById('text');
    const out   = document.getElementById('out');
    const spin  = document.getElementById('spin');
    const jsonEl= document.getElementById('json');

    function showAlert(msg, type="info") {
      out.className = 'alert alert-' + type;
      out.textContent = msg;
      out.classList.remove('d-none');
    }

    function renderSentiment(data) {
      const sentiment  = String((data && data.sentiment) || '').toLowerCase();
      const isPositive = sentiment === 'positive';
      const emoji      = isPositive ? '😊' : (sentiment === 'negative' ? '😠' : '🤔');

      out.className = 'alert ' + (isPositive ? 'alert-success' :
                                  (sentiment === 'negative' ? 'alert-danger' : 'alert-secondary'));
      out.hidden = false;
      const conf = (typeof data.confidence === 'number')
        ? `<div class="small text-muted">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>`
        : '';

      out.innerHTML = `
        <div class="d-flex align-items-center gap-2">
          <span style="font-size:1.5rem">${emoji}</span>
          <div>
            <div class="fw-semibold">
              Sentiment:
              <span class="badge ${isPositive ? 'bg-success' :
                                    (sentiment === 'negative' ? 'bg-danger' : 'bg-secondary')} text-uppercase">
                ${sentiment || 'unknown'}
              </span>
            </div>
            ${conf}
          </div>
        </div>
      `;

      // Optional: also show the raw JSON below
      jsonEl.textContent = JSON.stringify(data || {}, null, 2);
      jsonEl.classList.remove('d-none');
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      out.classList.add('d-none');
      jsonEl.classList.add('d-none');

      const text = ta.value.trim();
      if (!text) { showAlert('Please enter some text.', 'warning'); return; }

      btn.disabled = true; spin.classList.remove('d-none');
      try {
        const res = await fetch('/classify', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);

        renderSentiment(data);
      } catch (err) {
        showAlert('Error: ' + err.message, 'danger');
      } finally {
        btn.disabled = false; spin.classList.add('d-none');
      }
    });
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")

@app.post("/classify")
def classify_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        result = classify(text)  # your existing function returns a dict like {"sentiment": "...", "confidence": ...?}
        if not isinstance(result, dict):
            result = {"sentiment": str(result)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "classification_failed", "detail": str(e)}), 500

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
