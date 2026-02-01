"""
PDF Report Generator Module
Uses fpdf2 to create professional business reports from analysis results
"""

from fpdf import FPDF
import pandas as pd
import datetime
import os

class BusinessReportGenerator(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arabic_font = "/usr/share/fonts/truetype/noto/NotoKufiArabic-Regular.ttf"
        if os.path.exists(self.arabic_font):
            self.add_font("Arabic", "", self.arabic_font)
            self.has_arabic = True
        else:
            self.has_arabic = False

    def header(self):
        # Logo placeholder
        if self.has_arabic:
            self.set_font('Arabic', '', 15)
        else:
            self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'STRATEGIC INTELLIGENCE REPORT', 0, 1, 'C')
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 5, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def add_title_page(self, title, subtitle):
        self.add_page()
        self.set_y(100)
        if self.has_arabic:
            self.set_font('Arabic', '', 24)
            self.cell(0, 20, title, 0, 1, 'C')
            self.set_font('Arabic', '', 16)
            self.cell(0, 15, subtitle, 0, 1, 'C')
        else:
            self.set_font('helvetica', 'B', 24)
            self.cell(0, 20, title, 0, 1, 'C')
            self.set_font('helvetica', '', 16)
            self.cell(0, 15, subtitle, 0, 1, 'C')
        
    def add_section_header(self, title):
        self.ln(10)
        if self.has_arabic:
            self.set_font('Arabic', '', 14)
        else:
            self.set_font('helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(5)

    def add_bullet_point(self, text):
        if self.has_arabic:
            self.set_font('Arabic', '', 11)
            self.multi_cell(0, 7, f" - {text}", 0, 'L')
        else:
            self.set_font('helvetica', '', 11)
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 7, f" - {safe_text}", 0, 'L')
        self.ln(2)

    def add_paragraph(self, text):
        if self.has_arabic:
            self.set_font('Arabic', '', 11)
            self.multi_cell(0, 7, text, 0, 'L')
        else:
            self.set_font('helvetica', '', 11)
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 7, safe_text, 0, 'L')
        self.ln(2)

    def add_data_table(self, df, title=None):
        if title:
            if self.has_arabic:
                self.set_font('Arabic', '', 12)
            else:
                self.set_font('helvetica', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
        
        # Limit columns for PDF layout
        df_display = df.iloc[:10, :5]
        
        if self.has_arabic:
            self.set_font('Arabic', '', 9)
        else:
            self.set_font('helvetica', 'B', 9)
        col_width = self.epw / len(df_display.columns)
        
        # Header
        for col in df_display.columns:
            self.cell(col_width, 7, str(col), 1, 0, 'C')
        self.ln()
        
        # Rows
        if self.has_arabic:
            self.set_font('Arabic', '', 8)
        else:
            self.set_font('helvetica', '', 8)
        for _, row in df_display.iterrows():
            for val in row:
                self.cell(col_width, 6, str(val)[:20], 1, 0, 'C')
            self.ln()
        self.ln(5)

def generate_strategic_pdf(report_data, output_path):
    """
    Main function to generate the strategic PDF
    report_data should contain:
    - title
    - executive_summary
    - kpis (list of dicts)
    - insights (list of dicts)
    - recommendations (list of dicts)
    - chat_summary (optional)
    """
    pdf = BusinessReportGenerator()
    pdf.alias_nb_pages()
    
    # Title Page
    pdf.add_title_page(
        report_data.get('title', 'Analysis Report'),
        report_data.get('subtitle', 'Strategic Intelligence Insights')
    )
    
    # Executive Summary
    if 'executive_summary' in report_data:
        pdf.add_page()
        pdf.add_section_header('EXECUTIVE SUMMARY')
        pdf.add_paragraph(report_data['executive_summary'])
    
    # KPIs
    if 'kpis' in report_data:
        pdf.add_section_header('KEY PERFORMANCE INDICATORS')
        for kpi in report_data['kpis']:
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 7, kpi.get('name', 'KPI'), 0, 1, 'L')
            pdf.set_font('helvetica', '', 11)
            pdf.cell(0, 7, f"Value: {kpi.get('value', 'N/A')}", 0, 1, 'L')
            pdf.ln(2)
            
    # Insights
    if 'insights' in report_data:
        pdf.add_section_header('STRATEGIC INSIGHTS')
        for insight in report_data['insights']:
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 7, insight.get('title', 'Insight'), 0, 1, 'L')
            pdf.set_font('helvetica', '', 11)
            pdf.multi_cell(0, 7, insight.get('finding', ''), 0, 'L')
            pdf.ln(2)
            
    # Recommendations
    if 'recommendations' in report_data:
        pdf.add_section_header('ACTIONABLE RECOMMENDATIONS')
        for rec in report_data['recommendations']:
            pdf.set_font('helvetica', 'B', 11)
            title = f"[{rec.get('priority', 'MED').upper()}] {rec.get('title', 'Recommendation')}"
            pdf.cell(0, 7, title, 0, 1, 'L')
            pdf.set_font('helvetica', '', 11)
            pdf.multi_cell(0, 7, rec.get('description', ''), 0, 'L')
            pdf.ln(2)
            
    # Chat History Summary
    if 'chat_summary' in report_data:
        pdf.add_page()
        pdf.add_section_header('CHATBOT Q&A SUMMARY')
        pdf.add_paragraph(report_data['chat_summary'])
        
    pdf.output(output_path)
    return output_path
