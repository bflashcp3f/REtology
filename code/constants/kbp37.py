
NO_RELATION = "no_relation"

LABEL_TO_ID = {'no_relation': 0, 'org:alternate_names(e1,e2)': 1, 'org:alternate_names(e2,e1)': 2, 'org:city_of_headquarters(e1,e2)': 3, 'org:city_of_headquarters(e2,e1)': 4, 'org:country_of_headquarters(e1,e2)': 5,
 'org:country_of_headquarters(e2,e1)': 6, 'org:founded(e1,e2)': 7, 'org:founded(e2,e1)': 8, 'org:founded_by(e1,e2)': 9, 'org:founded_by(e2,e1)': 10, 'org:members(e1,e2)': 11, 'org:members(e2,e1)': 12,
 'org:stateorprovince_of_headquarters(e1,e2)': 13, 'org:stateorprovince_of_headquarters(e2,e1)': 14, 'org:subsidiaries(e1,e2)': 15, 'org:subsidiaries(e2,e1)': 16, 'org:top_members/employees(e1,e2)': 17,
 'org:top_members/employees(e2,e1)': 18, 'per:alternate_names(e1,e2)': 19, 'per:alternate_names(e2,e1)': 20, 'per:cities_of_residence(e1,e2)': 21, 'per:cities_of_residence(e2,e1)': 22, 'per:countries_of_residence(e1,e2)': 23,
 'per:countries_of_residence(e2,e1)': 24, 'per:country_of_birth(e1,e2)': 25, 'per:country_of_birth(e2,e1)': 26, 'per:employee_of(e1,e2)': 27, 'per:employee_of(e2,e1)': 28, 'per:origin(e1,e2)': 29, 'per:origin(e2,e1)': 30,
 'per:spouse(e1,e2)': 31, 'per:spouse(e2,e1)': 32, 'per:stateorprovinces_of_residence(e1,e2)': 33, 'per:stateorprovinces_of_residence(e2,e1)': 34, 'per:title(e1,e2)': 35, 'per:title(e2,e1)': 36}

ID_TO_LABEL = dict([(val, key) for key, val in LABEL_TO_ID.items()])

GRAMMER_NER_START = ["<e1>", "<e2>"]
GRAMMER_NER_END = ["</e1>", "</e2>"]