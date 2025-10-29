from typing import List

from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)


class MedsToCehrbertOMOP_Transforms(MedsToCehrBertConversion):

    def __init__(self, **kwargs):
        self.disconnect_problem_list_events = kwargs.get("disconnect_problem_list_events", False)
        super().__init__(**kwargs)

    def _create_visit_matching_rules(self) -> List[str]:
        return ["Visit/"]

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return ["Visit/ER"]

    def _create_admission_matching_rules(self) -> List[str]:
        return [
            "Visit//IP//start",
            "Visit//ERIP//start",
            "Visit//ER//start",
            "CMS Place of Service//51//start",
            "CMS Place of Service//61//start",
            "CMS Place of Service//20//start",
            "CMS Place of Service//15//start",
            "CMS Place of Service//02//start" ]

    def _create_discharge_matching_rules(self) -> List[str]:
        return [
            "CMS Place of Service//12",
            "CMS Place of Service//31",
            "SNOMED//371827001",
            "PCORNet//Generic-NI",
            "SNOMED//397709008",
            "Medicare Specialty//A4",
            "CMS Place of Service//21",
            "CMS Place of Service//61",
            "CMS Place of Service//51",
            "SNOMED//225928004",
            "CMS Place of Service//34",
            "PCORNet//Generic-OT",
            "CMS Place of Service//27",
            "CMS Place of Service//33",
            "CMS Place of Service//09",
            "CMS Place of Service//32",
            # ER codes
            "CMS Place of Service//12",
            "PCORNet//Generic-NI",
            "SNOMED//371827001",
            "CMS Place of Service//21",
            "NUCC//261Q00000X",
            "SNOMED//225928004",
            "CMS Place of Service//51",
            "CMS Place of Service//23",
            "CMS Place of Service//31",
            "CMS Place of Service//34",
            "SNOMED//397709008",
            "CMS Place of Service//24",
            "SNOMED//34596002",
            "PCORNet//Generic-OT",
            "CMS Place of Service//09",
            "Visit//ER//end",
            "Visit//ERIP//end",
            "Visit//IP//end",
            "CMS Place of Service//20//end",
            "CMS Place of Service//15//end",
            "CMS Place of Service//02//end",
        ]

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        return []
