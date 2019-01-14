class Sample:
    CONTROL = '1'
    CASE = '2'
    ONE_HOT_ENCODING = True
    ENCODING = ["1,0", "1,1", "0,1", "0,0"]

    def __init__(self, phenotype):
        self.phenotype = str(phenotype)
        self.genotype_data = ''

    def is_control(self):
        return self.phenotype == self.CONTROL

    def is_case(self):
        return self.phenotype == self.CASE

    def extend_genotype_data(self, genotype):
        converted_data = int(genotype, 2)
        if self.ONE_HOT_ENCODING:
            self.genotype_data += ',{}'.format(self.ENCODING[converted_data])
        else:
            self.genotype_data += ',{:n}'.format(converted_data)

    def print_me_for_ANN(self):
        phenotype = '0' if self.is_control() else '1' if self.is_case() else ''
        return '{}{}'.format(phenotype, self.genotype_data)