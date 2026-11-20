"""P3: grupos de validación cruzada y ausencia de fuga entre pacientes."""

import pytest

from src.training import cv
from src.training import dataset_loader as dl


def test_p3_each_ds1_record_exactly_once():
    flat = [r for f in cv.CV_FOLDS for r in f]
    assert sorted(flat) == sorted(dl.DS1) and len(flat) == len(set(flat)) == 22
    assert len(cv.CV_FOLDS) == 5


def test_p3_no_test_or_paced_records_in_folds():
    flat = {r for f in cv.CV_FOLDS for r in f}
    assert not flat & set(dl.DS2) and not flat & set(dl.PACED)


def test_p3_structure_check_passes_and_reports_known_leak():
    notes = cv.check_fold_structure()
    assert any("201" in n and "202" in n for n in notes)


@pytest.mark.parametrize("bad", [
    cv.CV_FOLDS[:-1],                                     # falta un grupo
    cv.CV_FOLDS + ((101,),),                              # registro repetido
    (cv.CV_FOLDS[0] + (100,),) + cv.CV_FOLDS[1:],         # registro de DS2
])
def test_p3_invalid_folds_are_rejected(bad):
    with pytest.raises(cv.FoldError):
        cv.check_fold_structure(bad)


def test_p3_same_patient_pairs_cannot_be_split(monkeypatch):
    monkeypatch.setattr(dl, "SAME_PATIENT", ((101, 106),))  # par ficticio en el mismo grupo: OK
    cv.check_fold_structure()
    monkeypatch.setattr(dl, "SAME_PATIENT", ((101, 108),))  # en grupos distintos: error
    with pytest.raises(cv.FoldError):
        cv.check_fold_structure()


def test_p3_inter_patient_split_untouched():
    assert not set(dl.DS1) & set(dl.DS2)
