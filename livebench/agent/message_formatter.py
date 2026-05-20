"""
Helper functions for formatting tool results into proper message format
"""

import base64
from typing import Dict, Any, List, Union


def format_result_for_logging(result: Any) -> str:
    """Format tool result for logging (handles binary data gracefully)"""
    if isinstance(result, dict):
        result_type = result.get('type', '')
        
        # Handle image-based results - don't log binary data
        if result_type in ['pdf_images', 'pptx_images']:
            images = result.get('images', [])
            image_count = len(images)
            if result_type == 'pdf_images':
                approx_pages = result.get('approximate_pages', image_count * 4)
                return f"{{'type': 'pdf_images', 'image_count': {image_count}, 'approximate_pages': {approx_pages}, 'message': 'PDF loaded successfully (binary data omitted from log)'}}"
            else:
                slide_count = result.get('slide_count', image_count)
                return f"{{'type': 'pptx_images', 'image_count': {image_count}, 'slide_count': {slide_count}, 'message': 'PowerPoint loaded successfully (binary data omitted from log)'}}"
        elif result_type == 'image':
            return "{'type': 'image', 'message': 'Image loaded successfully (binary data omitted from log)'}"
    
    # For non-binary results, return string representation
    result_str = str(result)
    # Truncate very long results
    if len(result_str) > 1000:
        return result_str[:1000] + "... (truncated)"
    return result_str


def format_tool_result_message(
    tool_name: str,
    tool_result: Any,
    tool_args: Dict,
    activity_completed: bool
) -> Dict[str, Union[str, List[Dict]]]:
    """Format tool result into proper message format"""
    if isinstance(tool_result, dict):
        result_type = tool_result.get('type', '')
        
        if result_type in ['pdf_images', 'pptx_images']:
            return _format_multimodal_message(tool_name, tool_result, activity_completed)
        elif result_type == 'image':
            return _format_image_message(tool_name, tool_result, activity_completed)
    
    return _format_text_message(tool_name, tool_result, tool_args, activity_completed)


def _format_multimodal_message(
    tool_name: str,
    tool_result: Dict,
    activity_completed: bool
) -> Dict[str, List[Dict]]:
    """Format PDF/PPTX images as multimodal message"""
    result_type = tool_result.get('type', '')
    images = tool_result.get('images', [])
    
    if result_type == 'pdf_images':
        image_count = tool_result.get('image_count', len(images))
        approx_pages = tool_result.get('approximate_pages', image_count * 4)
        text_summary = f"Tool result: Successfully read PDF file.\nLoaded ~{approx_pages} pages as {image_count} combined images."
    elif result_type == 'pptx_images':
        slide_count = tool_result.get('slide_count', len(images))
        text_summary = f"Tool result: Successfully read PowerPoint file.\nLoaded {slide_count} slides."
    else:
        text_summary = f"Tool result: Loaded {len(images)} images."
    
    if activity_completed:
        text_summary += "\n\nGreat! You completed your daily activity."
    
    content = [{"type": "text", "text": text_summary}]
    
    for img_bytes in images:
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })
    
    return {"role": "user", "content": content}


def _format_image_message(
    tool_name: str,
    tool_result: Dict,
    activity_completed: bool
) -> Dict[str, List[Dict]]:
    """Format single image as multimodal message"""
    image_data = tool_result.get('image_data', '')
    text_summary = "Tool result: Successfully read image file."
    
    if activity_completed:
        text_summary += "\n\nGreat! You completed your daily activity."
    
    content = [
        {"type": "text", "text": text_summary},
        {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}}
    ]
    
    return {"role": "user", "content": content}


_MAX_STDOUT = 3000
_MAX_STDERR = 2000
_MAX_TOTAL  = 6000


def _truncate_tool_result(result: Any) -> Any:
    """Truncate stdout/stderr fields so model context stays manageable."""
    import json as _json
    if not isinstance(result, dict):
        s = str(result)
        return s[:_MAX_TOTAL] + "... [truncated]" if len(s) > _MAX_TOTAL else s

    r = dict(result)
    if "stdout" in r and isinstance(r["stdout"], str) and len(r["stdout"]) > _MAX_STDOUT:
        r["stdout"] = r["stdout"][:_MAX_STDOUT] + "... [truncated]"
    if "stderr" in r and isinstance(r["stderr"], str) and len(r["stderr"]) > _MAX_STDERR:
        r["stderr"] = r["stderr"][:_MAX_STDERR] + "... [truncated]"
    return r


def _format_text_message(
    tool_name: str,
    tool_result: Any,
    tool_args: Dict,
    activity_completed: bool
) -> Dict[str, str]:
    """Format regular text tool result"""
    import json as _json
    truncated = _truncate_tool_result(tool_result)
    if isinstance(truncated, (dict, list)):
        result_str = _json.dumps(truncated, ensure_ascii=False)
    else:
        result_str = str(truncated)
    tool_result_message = f"Tool result: {result_str}"
    
    if tool_name == 'decide_activity' and 'work' in str(tool_args).lower():
        tool_result_message += "\n\nYou decided to WORK. Complete it now!"
    elif tool_name == 'decide_activity' and 'learn' in str(tool_args).lower():
        tool_result_message += "\n\nYou decided to LEARN. Complete it now!"
    elif activity_completed:
        tool_result_message += "\n\nGreat! You completed your daily activity."
    
    return {"role": "user", "content": tool_result_message}
