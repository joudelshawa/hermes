# tests/test_report_generator.py
from agents.report_generator import ReportGenerator

def test_report_extraction():
    agent = ReportGenerator()
    text = agent.extract_text("sample.pdf")
    assert isinstance(text, str)
