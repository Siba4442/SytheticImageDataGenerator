import gradio as gr

# Simple mapping - just 2 characters for testing
SIMPLE_MAPPING = {
    'k': 'କ',  # ka
    'a': 'ା',  # aa vowel
}

def create_keyboard_html():
    """Create HTML with JavaScript for direct physical keyboard input"""
    return """
    <div style="padding: 10px; background: #f5f5f5; border-radius: 8px; margin: 10px 0;">
        <h4>Physical Keyboard Test</h4>
        <p><strong>k</strong> → <strong>କ</strong> &nbsp;&nbsp; <strong>a</strong> → <strong>ା</strong></p>
        <p style="color: #666; font-size: 12px;">Click in the text box below and type 'k' or 'a' on your keyboard</p>
    </div>
    
    <script>
        const mapping = {"k": "କ", "a": "ା"};
        
        document.addEventListener('keydown', function(event) {{
            // Find active textarea in Gradio
            const textareas = document.querySelectorAll('textarea');
            let activeTextarea = null;
            
            for (let textarea of textareas) {
                if (document.activeElement === textarea) {{
                    activeTextarea = textarea;
                    break;
                }
            }
            
            if (!activeTextarea) return;
            
            // Skip special keys
            if (event.ctrlKey || event.altKey || event.metaKey || 
                ['Backspace', 'Enter', 'Tab', 'ArrowLeft', 'ArrowRight', 'Delete'].includes(event.key)) {
                return;
            }
            
            // Check if key has Odia mapping
            const odiaChar = mapping[event.key.toLowerCase()];
            if (odiaChar) {
                event.preventDefault();
                
                // Insert Odia character
                const start = activeTextarea.selectionStart;
                const end = activeTextarea.selectionEnd;
                activeTextarea.value = activeTextarea.value.substring(0, start) + odiaChar + activeTextarea.value.substring(end);
                activeTextarea.selectionStart = activeTextarea.selectionEnd = start + odiaChar.length;
                activeTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
    </script>
    """

# Create Gradio interface
with gr.Blocks(title="Physical Keyboard Test") as demo:
    gr.Markdown("# Physical Keyboard → Odia Test")
    
    keyboard_html = gr.HTML(create_keyboard_html())
    
    textbox = gr.Textbox(
        label="Click here and type 'k' or 'a'",
        lines=3,
        placeholder="Click here first, then use your keyboard..."
    )
    
    gr.Markdown("**Try typing:** ka, ak, kaa, etc.")

if __name__ == "__main__":
    demo.launch()