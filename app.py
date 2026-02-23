import gradio as gr

def greet(name):
    return "Hello " + name + "!!. Testing Hugging Face deployment!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
