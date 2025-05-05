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

## Files to Remove

The following files are no longer needed and can be safely removed:

```
combined_tabbed_backup.py
combined_tabbed_backup_copy.py
combined_tabbed_tail.txt
debug_txt_*.txt
debuglog.md
enable_lm_studio_vision.py
fix_image_upload.py
fix_lm_studio.py
hybrid_fix.py
lm_studio_dropdown_integration.py
lm_studio_dropdown_patch.txt
lm_studio_dropdown_revised_patch.txt
RAG_modify.py
simple_fix.py
test_lm_studio_dropdown.py
test_lm_studio_dropdown_fixed.py
test_lm_studio_vision.py
```

## Folders to Remove

The following folders are not needed for the current system:

```
archieved/
temp_backup/
__pycache__/
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
