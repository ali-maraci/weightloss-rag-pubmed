"""Tests for app/core/safety.py SafetyClassifier."""
import os
os.environ.setdefault("NCBI_API_KEY", "test-key")
os.environ.setdefault("NCBI_EMAIL", "test@example.com")

from app.core.safety import SafetyClassifier, SafetyLevel


class TestSafetyClassifier:
    def setup_method(self):
        self.classifier = SafetyClassifier()

    def test_dosage_query_is_high_risk(self):
        assert self.classifier.classify("What dose of semaglutide should I take?") == SafetyLevel.HIGH

    def test_drug_interaction_is_high_risk(self):
        assert self.classifier.classify("Can I combine semaglutide with insulin together?") == SafetyLevel.HIGH

    def test_emergency_is_high_risk(self):
        assert self.classifier.classify("Is pancreatitis from Ozempic fatal?") == SafetyLevel.HIGH

    def test_side_effects_is_moderate(self):
        assert self.classifier.classify("What are the side effects of semaglutide?") == SafetyLevel.MODERATE

    def test_cancer_risk_is_moderate(self):
        assert self.classifier.classify("Is there a thyroid cancer risk?") == SafetyLevel.MODERATE

    def test_general_question_is_low(self):
        assert self.classifier.classify("What is semaglutide?") == SafetyLevel.LOW

    def test_mechanism_question_is_low(self):
        assert self.classifier.classify("How does GLP-1 work?") == SafetyLevel.LOW

    def test_high_risk_has_disclaimer(self):
        _, disclaimer = self.classifier.classify_and_disclaim("Should I take 2mg of semaglutide?")
        assert "NOT medical advice" in disclaimer
        assert "healthcare provider" in disclaimer

    def test_moderate_has_disclaimer(self):
        _, disclaimer = self.classifier.classify_and_disclaim("What are the side effects?")
        assert "Consult your healthcare provider" in disclaimer

    def test_low_has_no_disclaimer(self):
        _, disclaimer = self.classifier.classify_and_disclaim("What is semaglutide?")
        assert disclaimer == ""

    def test_pregnancy_is_high_risk(self):
        assert self.classifier.classify("Can I take Wegovy while pregnant?") == SafetyLevel.HIGH

    def test_should_i_stop_is_high_risk(self):
        assert self.classifier.classify("Should I stop taking Ozempic before surgery?") == SafetyLevel.HIGH
