# Multimodal Processing Guide

## Understanding Multimodal Inputs

This application supports three types of input modalities:

1. **Text**: Mandatory for all queries - used to provide instructions or questions
2. **Images**: Optional - can be uploaded to analyze visual content
3. **Videos**: Optional - can be uploaded for frame-by-frame analysis

## Input Requirements

### Text Input
- Always required for every query
- Provides the context or question for the system
- Can reference uploaded media ("What's in this image?") or be standalone

### Image Input
- Supported formats: JPG, PNG, GIF, BMP
- Recommended resolution: 512x512 to 1024x1024
- Max file size: 5MB

### Video Input
- Supported formats: MP4, MOV, AVI
- Recommended length: Under 2 minutes (longer videos may cause timeout)
- Max file size: 25MB
- Resolution: Up to 1080p

## Processing Modes

### Pure Chatbot Mode
When using the system without RAG (direct model analysis):
- Uses only the LLM's knowledge
- Response labeled as "Direct Model Analysis"
- No context retrieval from knowledge base
- Purely based on model capabilities

### RAG Mode
When using the system with knowledge base:
- Retrieves relevant context from your documents
- Combines retrieved knowledge with model capabilities
- Shows sources used in generating the response
- Better for domain-specific or technical questions

## Tips for Effective Use

### For Text Queries
- Be specific and clear in your questions
- Mention specific topics you want information about
- Use domain-specific terminology when appropriate

### For Image Analysis
- Use clear, well-lit images
- Center the subject of interest
- Provide specific questions about the image

### For Video Analysis
- Keep videos concise and focused
- Consider video quality (lighting, stability)
- Ask specific questions about the video content

## How Multimodal Processing Works

1. **Text Processing**:
   - Your text query is analyzed for intent
   - If in RAG mode, relevant documents are retrieved
   - If not in RAG mode, direct model analysis is performed

2. **Image Processing**:
   - Images are analyzed by the vision model
   - Visual features are extracted
   - Combined with text query for comprehensive understanding

3. **Video Processing**:
   - Key frames are extracted from the video
   - Each frame is analyzed separately
   - Analysis is aggregated for a cohesive understanding
   - Temporal relationships between frames are considered

## Performance Considerations

The system's performance depends on:
- Your hardware capabilities
- Input complexity
- Query type

For optimal results:
- Use high-quality but reasonably sized media files
- Be specific in your queries
- Start with simple queries before complex ones

Remember that text input is always required, while image and video inputs are optional.
