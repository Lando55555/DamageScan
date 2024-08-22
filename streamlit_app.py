import streamlit as st
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Damage Assessment Report Generator")

        with gr.Row():
            pdf_input = gr.File(label="Upload Damage Report PDF", file_count="single")

        with gr.Row():
            claim_number = gr.Textbox(label="Claim Number")

        with gr.Row():
            notes_input = gr.Textbox(label="Additional Notes (optional)", lines=3)

        with gr.Row():
            submit_btn = gr.Button("Process Report")

        with gr.Tabs():
            with gr.TabItem("Analysis Results"):
                weather_output = gr.JSON(label="Weather Data")
                e3_output = gr.JSON(label="E3 Calculations")
                scope_output = gr.Textbox(label="Scope of Work")
                cost_output = gr.JSON(label="Cost Assessment")
                image_analysis_output = gr.JSON(label="Photo Analysis Results")

            with gr.TabItem("Generated Report"):
                report_output = gr.File(label="Download Report")

            with gr.TabItem("Extracted Images"):
                image_gallery = gr.Gallery(label="Extracted Images")

        submit_btn.click(
            fn=process_and_analyze,
            inputs=[pdf_input, claim_number, notes_input],
            outputs=[weather_output, e3_output, scope_output, cost_output, image_analysis_output, report_output, image_gallery]
        )

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
