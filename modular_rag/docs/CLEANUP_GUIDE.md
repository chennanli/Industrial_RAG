# Project Cleanup Guide

This document provides guidance on how to clean up the project folder for a more organized repository.

## Essential Files and Folders

These files and folders are necessary for the system to function:

```
LLM_Project/Qwen2.5/
├── launch_rag.py                # Main launcher script
├── qwen25_env/                  # Python virtual environment
├── RAG_pdf/                     # PDF documents folder
├── frame_cache/                 # Video frame cache folder
└── modular_rag/                 # Main code directory
```

## Documentation Files to Keep

These files provide useful documentation and should be kept, preferably in the `docs/` folder:

```
docs/
├── README.md                    # Documentation overview
├── RAG_concise_readme.md        # Original README
├── OFFLOAD_FOLDER_EXPLANATION.md # Memory management docs
├── IMAGE_UPLOAD_FIX_GUIDE.md    # Image upload troubleshooting
└── LM_STUDIO_FIX_GUIDE.md       # LM Studio integration guide
```



## Project Organization for GitHub

When committing to GitHub, consider this structure:

```
LLM_Project/Qwen2.5/
├── launch_rag.py
├── modular_rag/                 # Main code
│   ├── app.py
│   ├── models/
│   ├── rag_modules/
│   ├── ui/
│   ├── utils/
│   └── docs/                    # Documentation folder
│       ├── README.md
│       ├── RAG_concise_readme.md
│       ├── OFFLOAD_FOLDER_EXPLANATION.md
│       ├── IMAGE_UPLOAD_FIX_GUIDE.md
│       ├── LM_STUDIO_FIX_GUIDE.md
│       └── CLEANUP_GUIDE.md
├── README.md                    # Main project README
├── .gitignore                   # Include patterns for qwen25_env, __pycache__, etc.
└── requirements.txt             # For environment reconstruction
```

## Creating a `.gitignore` File

To avoid committing unnecessary files to GitHub, create a `.gitignore` file with:

```
# Environment and cache
qwen25_env/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
env/
venv/
ENV/

# PDF and data files (optional, if you don't want to include your test PDFs)
RAG_pdf/*.pdf
frame_cache/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
```

This will ensure a clean, well-organized repository that others can easily understand and use.
