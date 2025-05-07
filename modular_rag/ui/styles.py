"""
CSS styles for the RAG UI
"""

# Define CSS styling for the UI
css = """
/* Force container to use full width of browser window */
.gradio-container {
    max-width: 100% !important; /* Override any theme default */
    width: 100% !important;
    margin: 0 auto !important;
}

/* Ensure content areas expand properly */
.main, .contain, .wrap, .wrap-inner, .container {
    max-width: 100% !important;
    width: 100% !important;
}

/* Override any theme constraints */
.gradio-app {
    width: 100% !important;
    max-width: none !important;
}
footer {
    display: none !important;
}
.footer-made-with-gradio {
    display: none !important;
}
.panel {
    background-color: #f7f2e5; /* Lighter background for more contrast */
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 20px; /* Increased bottom margin for more separation */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* Stronger shadow */
    border: 3px solid #3f3f3f; /* Darker, thicker border */
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
    color: #3f3f3f; /* Matched with border color */
    font-weight: bold;
    margin-bottom: 12px;
    font-size: 1.3em; /* Slightly larger */
    border-bottom: 2px solid #3f3f3f; /* Added border under headers */
    padding-bottom: 8px;
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
    margin-top: 15px;
    border: 2px solid #f8fafc; /* Matching light border with text for contrast */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Stronger shadow */
}
.source-box h3 {
    color: #f8fafc !important; /* Light text title */
    font-weight: bold;
    margin-bottom: 12px;
    border-bottom: 2px solid #f8fafc; /* Thicker, matching border */
    padding-bottom: 8px;
    font-size: 1.2em; /* Slightly larger */
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
.cancel-btn {
    background-color: #ef4444 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.cancel-btn:hover {
    background-color: #dc2626 !important;
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
