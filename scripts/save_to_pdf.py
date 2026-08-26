

import pdfkit


def save_client_pdf(shap_force_plot,plotly_fig,customer_id, score,output_path):
    shap_html = shap_force_plot.html()

    plotly_html = plotly_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Client {customer_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
            }}

            h1 {{
                margin-bottom: 10px;
            }}

            .score {{
                font-size: 20px;
                margin-bottom: 30px;
            }}

            .section {{
                margin-top: 30px;
                page-break-inside: avoid;
            }}
        </style>
    </head>

    <body>

        <h1>Local Interpretation — Client {customer_id}</h1>

        <div class="score">
            Default probability: <b>{score:.2%}</b>
        </div>

        <div class="section">
            <h2>SHAP Force Plot</h2>
            {shap_html}
        </div>

        <div class="section">
            <h2>Client Information and Comparison</h2>
            {plotly_html}
        </div>

    </body>
    </html>
    """

    pdfkit.from_string(
        html,
        output_path,
        options={
            "enable-javascript": "",
            "javascript-delay": "2000",
            "page-size": "A4",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
            "margin-left": "10mm",
            "margin-right": "10mm",
        }
    )

    print(f"PDF saved to: {output_path}")
