# --- Imports ---------------------------------------------------------------
# Flask: tiny web framework to serve one HTML page and a POST endpoint
from flask import Flask, request, jsonify, Response

# classify():  existing function that calls OpenAI and returns a dict
from openai_trained import classify

# --- App bootstrap ---------------------------------------------------------
app = Flask(__name__)

# --- GET / : serve a single HTML page -------------------------------------
@app.get("/")
def index():
    # We return the entire HTML document as a string. It includes:
    # - Bootstrap (for quick styling)
    # - A simple form (textarea + button)
    # - A result area to show a pretty badge + raw JSON
    # - A small JS script that calls POST /classify and renders the result
    html = """
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8"/>
  <title>Sentiment Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <!-- Bootstrap CSS via CDN for styling -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="py-5">
  <div class="container" style="max-width: 760px;">
    <h1 class="mb-4">Sentiment Classifier</h1>

    <!-- Card container for the form + results -->
    <div class="card shadow-sm">
      <div class="card-body">
        <!-- The input form. We handle submit with JavaScript (no page reload). -->
        <form id="form" class="mb-3">
          <div class="mb-3">
            <label for="text" class="form-label">Enter a comment</label>
            <textarea id="text" class="form-control" rows="4"
              placeholder="e.g. 'awful support, totally broken and late'"></textarea>
          </div>
          <button id="btn" type="submit" class="btn btn-primary">
            <!-- Small spinner shown while waiting for the server response -->
            <span class="spinner-border spinner-border-sm me-2 d-none" id="spin"></span>
            Classify
          </button>
        </form>

        <!-- Pretty, human-friendly result area (Bootstrap alert) -->
        <div id="out" class="alert d-none" role="alert"></div>

        <!-- Optional: show the raw JSON response below -->
        <pre id="json" class="bg-light p-3 rounded d-none"></pre>
      </div>
    </div>
  </div>

  <!-- Bootstrap JS bundle (for components; not strictly required here) -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <script>
    // --- Grab references to DOM elements we'll interact with ---------------
    const form  = document.getElementById('form');
    const btn   = document.getElementById('btn');
    const ta    = document.getElementById('text');
    const out   = document.getElementById('out');
    const spin  = document.getElementById('spin');
    const jsonEl= document.getElementById('json');

    // Helper to show a Bootstrap alert with a message
    function showAlert(msg, type="info") {
      out.className = 'alert alert-' + type;
      out.textContent = msg;
      out.classList.remove('d-none');
    }

    // Pretty renderer: shows emoji + colored badge; also prints raw JSON below
    function renderSentiment(data) {
      const sentiment  = String((data && data.sentiment) || '').toLowerCase();
      const isPositive = sentiment === 'positive';
      const emoji      = isPositive ? '😊' : (sentiment === 'negative' ? '😠' : '🤔');

      // Pick alert color based on sentiment (green for positive, red for negative)
      out.className = 'alert ' + (isPositive ? 'alert-success' :
                                  (sentiment === 'negative' ? 'alert-danger' : 'alert-secondary'));
      out.hidden = false;

      // If your classify() returns a numeric confidence, show it
      const conf = (typeof data.confidence === 'number')
        ? `<div class="small text-muted">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>`
        : '';

      // Compose the pretty HTML
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

      // Also show the raw JSON (useful for debugging)
      jsonEl.textContent = JSON.stringify(data || {}, null, 2);
      jsonEl.classList.remove('d-none');
    }

    // --- Form submit handler: call POST /classify with fetch ----------------
    form.addEventListener('submit', async (e) => {
      e.preventDefault();                 // stop the default form POST/reload
      out.classList.add('d-none');        // hide previous results
      jsonEl.classList.add('d-none');

      const text = ta.value.trim();       // get user input
      if (!text) { showAlert('Please enter some text.', 'warning'); return; }

      // Disable the button + show spinner while calling the backend
      btn.disabled = true; spin.classList.remove('d-none');
      try {
        // Send JSON {text: "..."} to our Flask endpoint
        const res = await fetch('/classify', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text })
        });

        // Parse JSON either way so we can show helpful errors
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);

        // Render a pretty result
        renderSentiment(data);
      } catch (err) {
        // Show any error message in a red alert
        showAlert('Error: ' + err.message, 'danger');
      } finally {
        // Re-enable the button and hide the spinner
        btn.disabled = false; spin.classList.add('d-none');
      }
    });
  </script>
</body>
</html>
"""
    # Tell Flask we’re returning HTML
    return Response(html, mimetype="text/html")


# --- POST /classify : server endpoint the JS calls -------------------------
@app.post("/classify")
def classify_route():
    # Safely read JSON body like {"text": "..."}
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    # Basic input validation
    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        # Call your classifier (must return a dict, e.g., {"sentiment": "positive", "confidence": 0.95})
        result = classify(text)

        # Defensive: if someone changes classify() to return a string, wrap it
        if not isinstance(result, dict):
            result = {"sentiment": str(result)}

        # Send JSON back to the browser
        return jsonify(result)

    except Exception as e:
        # Any exception becomes a 500 with a short message
        return jsonify({"error": "classification_failed", "detail": str(e)}), 500


# --- Local dev entrypoint ---------------------------------------------------
if __name__ == "__main__":
    # Start Flask dev server on http://127.0.0.1:5000 (auto-reload enabled in debug mode)
    app.run(host="127.0.0.1", port=5000, debug=True)
