"""
CSS styles for the RAG UI
"""

# Define CSS styling for the UI
css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
footer {
    display: none !important;
}
.footer-made-with-gradio {
    display: none !important;
}
.panel {
    background-color: #fffaf0; /* Light warm solar color */
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
}
.status-ready {
    background-color: #10b981;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.status-processing {
    background-color: #3b82f6;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.status-error {
    background-color: #ef4444;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.header-text {
    color: #111827;
    font-weight: bold;
    margin-bottom: 12px;
    font-size: 1.2em;
}
.pure-text-tab, .image-tab, .video-tab {
    min-height: 40px;
    padding: 10px;
    margin-top: 10px;
}
.query-box {
    min-height: 100px;
    border-radius: 8px;
    border: 1px solid #374151;
}
.result-box {
    height: 350px;
    min-height: 150px;
    background-color: #ffffff; /* White background */
    color: #111827 !important; /* Dark text color, forced */
    padding: 20px;
    border-radius: 8px;
    overflow-y: auto !important;
    display: block;
    border: 1px solid #e5e7eb;
    line-height: 1.6;
}
.result-box p, .result-box li, .result-box span, .result-box div {
    color: #111827 !important; /* Ensure child elements also inherit text color */
}
.result-box h1, .result-box h2, .result-box h3, .result-box h4, .result-box h5, .result-box h6 {
     color: #000000 !important; /* Black color for headings */
     margin-top: 16px;
     margin-bottom: 8px;
     font-weight: bold;
}
.source-box {
    background-color: #1e293b; /* Dark background for high contrast */
    color: #f8fafc !important; /* Light text for high contrast */ 
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    border: 1px solid #334155;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
}
.source-box h3 {
    color: #f8fafc !important; /* Light text title */
    font-weight: bold;
    margin-bottom: 12px;
    border-bottom: 1px solid #475569;
    padding-bottom: 4px;
}
.source-box * {
    color: #f8fafc !important; /* Ensure all content is light */
}
.submit-btn {
    background-color: #3b82f6 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.submit-btn:hover {
    background-color: #2563eb !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.result-box h2 { /* Target headings within the markdown output */
    margin-top: 16px;
    margin-bottom: 8px;
    font-weight: bold;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}
.result-box p { /* Target paragraphs within the markdown output */
    margin-bottom: 12px;
}
.result-box ul, .result-box ol {
    margin-left: 20px;
    margin-bottom: 12px;
}
.result-box li {
    margin-bottom: 4px;
}
.result-box code {
    background-color: #f3f4f6;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: monospace;
    color: #111827 !important;
}
.result-box pre {
    background-color: #f3f4f6;
    padding: 10px;
    border-radius: 8px;
    overflow-x: auto;
    margin-bottom: 12px;
}
.init-btn {
    background-color: #10b981 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.init-btn:hover {
    background-color: #059669 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
body {
    background-color: #fff8e1; /* Light warm background for the entire page */
}
/* Style for model selection panel */
.model-panel {
    background-color: #f3f4f6; 
    padding: 12px; 
    border-radius: 8px;
    border: 1px solid #d1d5db;
    margin-bottom: 15px;
}
.refresh-btn {
    background-color: #4b5563 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 5px 10px !important;
    transition: all 0.2s ease;
}
.refresh-btn:hover {
    background-color: #374151 !important;
    transform: translateY(-1px);
}
.model-status {
    font-size: 0.9em;
    font-style: italic;
    color: #4b5563;
    margin-top: 5px;
}
"""
