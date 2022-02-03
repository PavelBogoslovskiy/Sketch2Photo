import gradio as gr
from predict import image_classifier


if __name__ == "__main__":
	gr.Interface(fn=image_classifier, inputs="image", outputs="image").launch(debug=True)