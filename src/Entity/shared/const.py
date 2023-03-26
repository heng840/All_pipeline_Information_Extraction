task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'WikiEvents': ["TTL", "INF", "VAL", "LOC", "COM", "CRM", "ORG", "FAC", "MHI", "WEA", "VEH", "SID", "ABS", "MON",
                   "GPE", "BOD", "PER"],
    'ACE': ['VEH:Water',
            'GPE:Nation',
            'ORG:Commercial',
            'GPE:State-or-Province',
            'Contact-Info:E-Mail',
            'Crime',
            'ORG:Non-Governmental',
            'Contact-Info:URL',
            'Sentence',
            'ORG:Religious',
            'VEH:Underspecified',
            'WEA:Projectile',
            'FAC:Building-Grounds',
            'PER:Group',
            'WEA:Exploding',
            'WEA:Biological',
            'Contact-Info:Phone-Number',
            'WEA:Chemical',
            'LOC:Land-Region-Natural',
            'WEA:Nuclear',
            'LOC:Region-General',
            'PER:Individual',
            'WEA:Sharp',
            'ORG:Sports',
            'ORG:Government',
            'ORG:Media',
            'LOC:Address',
            'WEA:Shooting',
            'LOC:Water-Body',
            'LOC:Boundary',
            'GPE:Population-Center',
            'GPE:Special',
            'LOC:Celestial',
            'FAC:Subarea-Facility',
            'PER:Indeterminate',
            'VEH:Subarea-Vehicle',
            'WEA:Blunt',
            'VEH:Land',
            'TIM:time',
            'Numeric:Money',
            'FAC:Airport',
            'GPE:GPE-Cluster',
            'ORG:Educational',
            'Job-Title',
            'GPE:County-or-District',
            'ORG:Entertainment',
            'Numeric:Percent',
            'LOC:Region-International',
            'WEA:Underspecified',
            'VEH:Air',
            'FAC:Path',
            'ORG:Medical-Science',
            'FAC:Plant',
            'GPE:Continent']
}



def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label