

NO_RELATION = "no_relation"

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4,
               'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8,
               'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14,
               'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19,
               'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25,
               'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30,
               'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35,
               'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40,
               'per:country_of_death': 41}
ID_TO_LABEL = dict([(val, key) for key, val in LABEL_TO_ID.items()])

# PAD_TOKEN = '<PAD>'
# UNK_TOKEN = '<UNK>'

SUBJ_NER_TO_ID = {'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11,
                 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

# GRAMMER_NER_START = []

#
# for item in SUBJ_NER_TO_ID.keys():
#     GRAMMER_NER_START.append("["+"SUBJ-"+item+"-START]")
#
# for item in OBJ_NER_TO_ID.keys():
#     GRAMMER_NER_START.append("["+"OBJ-"+item+"-START]")
#
#
# GRAMMER_NER_END = []
#
# for item in SUBJ_NER_TO_ID.keys():
#     GRAMMER_NER_END.append("["+"SUBJ-"+item+"-END]")
#
# for item in OBJ_NER_TO_ID.keys():
#     GRAMMER_NER_END.append("["+"OBJ-"+item+"-END]")

GRAMMER_NER_START = ["[subj-start]", "[obj-start]"]

for item in SUBJ_NER_TO_ID.keys():
    GRAMMER_NER_START.append("[" + "subj-" + item.lower() + "-start]")

for item in OBJ_NER_TO_ID.keys():
    GRAMMER_NER_START.append("[" + "obj-" + item.lower() + "-start]")

GRAMMER_NER_END = ["[subj-end]", "[obj-end]"]

for item in SUBJ_NER_TO_ID.keys():
    GRAMMER_NER_END.append("[" + "subj-" + item.lower() + "-end]")

for item in OBJ_NER_TO_ID.keys():
    GRAMMER_NER_END.append("[" + "obj-" + item.lower() + "-end]")
