class SNP:
    def __init__(self):
        self.bim_pos = 0
        self.affy_name = ''
        self.rsid = ''

    def set_affy_name(self, affy_name):
        self.affy_name = affy_name

    def get_affy_name(self):
        return self.affy_name

    def set_rsid(self, rsid):
        self.rsid = rsid

    def set_bim_pos(self, bim_pos):
        self.bim_pos = int(bim_pos)

    def get_bim_pos(self):
        return self.bim_pos