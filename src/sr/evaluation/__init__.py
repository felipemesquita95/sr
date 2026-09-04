"""Subsistema de avaliação: métricas, figuras e relatórios."""

from sr.evaluation.metrics import EvaluationResult, evaluate_model
from sr.evaluation.report import write_fold_report, write_summary

__all__ = ['EvaluationResult', 'evaluate_model', 'write_fold_report', 'write_summary']
