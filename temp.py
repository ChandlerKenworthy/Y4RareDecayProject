import ROOT
# from AnalysisPython.PyRoUts import hID, cpp, VE

# from lib import utils
from lib import pdg
from lib.model import AbstractModel

from IPython import embed as shell  # noqa

from AnalysisPython.PyRoUts import cpp
PolyPositive = cpp.Analysis.Models.PolyPositive
ExpoPositive = cpp.Analysis.Models.ExpoPositive
CrystalBallDS = cpp.Analysis.Models.CrystalBallDS


class Background(object):

    def __init__(self, dm, dm_begin, dm_end, order):

        self.tau = ROOT.RooRealVar("exp_tau", "exp_tau", -1.5, -20, 1)
        self.phi1 = ROOT.RooRealVar(
            "poly_phi1", "poly_phi1", 0, -3.1415, 3.1415)
        self.phi2 = ROOT.RooRealVar(
            "poly_phi2", "poly_phi2", 0, -3.1415, 3.1415)
        self.phi3 = ROOT.RooRealVar(
            "poly_phi3", "poly_phi3", 0, -3.1415, 3.1415)
        self.phi4 = ROOT.RooRealVar(
            "poly_phi4", "poly_phi4", 0, -3.1415, 3.1415)
        self.phi5 = ROOT.RooRealVar(
            "poly_phi5", "poly_phi5", 0, -3.1415, 3.1415)
        self.phi6 = ROOT.RooRealVar(
            "poly_phi6", "poly_phi6", 0, -3.1415, 3.1415)
        self.phi7 = ROOT.RooRealVar(
            "poly_phi7", "poly_phi7", 0, -3.1415, 3.1415)

        self.alist = ROOT.RooArgList()
        for i in range(order):
            self.alist.add(getattr(self, "phi%d" % (i + 1)))

        self.exp_pdf = ExpoPositive(
            "exp_pdf", "exp_pdf",
            dm, self.tau,
            self.alist,
            dm_begin,
            dm_end)

        self.pdf = self.exp_pdf


class Chib1P(object):

    def __init__(self, dm, frac):
        dm_pdg = pdg.DM1S[0].value()
        self.mean1 = ROOT.RooRealVar("mean_b1_1p", "mean_b1_1p", dm_pdg,
                                     dm_pdg - 0.05, dm_pdg + 0.05)

        diff = pdg.DM1S[1].value() - pdg.DM1S[0].value()
        self.dmb2b1 = ROOT.RooRealVar("dmb2b1_1p", "dmb2b1_1p", diff)
        self.dmb2b1.setConstant(True)

        alistb2b1 = ROOT.RooArgList(self.mean1, self.dmb2b1)
        self.mean2 = ROOT.RooFormulaVar("mean_b2_1p", "mean_b2_1p",
                                        "%s+%s" % (self.mean1.GetName(),
                                                   self.dmb2b1.GetName()
                                                   ),
                                        alistb2b1)

        self.sigma = ROOT.RooRealVar("sigma1", "sigma1",
                                     0.024,
                                     0.024 - 0.01,
                                     0.024 + 0.01)

        self.nR = ROOT.RooRealVar("nr1", "nr1", 5, 1, 5)
        self.alphaR = ROOT.RooRealVar(
            "ar1", "ar1", -0.2, -5, 0)  # -1.23
        self.alphaR.setConstant(True)

        self.nL = ROOT.RooRealVar("nl1", "nl1", 1, 1, 5)
        self.nL.setConstant(True)
        self.alphaL = ROOT.RooRealVar("al1", "al1", 0.7, 0, 5)  # -1.23
        self.alphaL.setConstant(True)

        self.pdf1 = CrystalBallDS("cb1_1", "cb1_1", dm, self.mean1,
                                  self.sigma, self.alphaL, self. nL,
                                  self.alphaR, self. nL)
        
        self.pdf2 = CrystalBallDS("cb2_1", "cb2_1", dm, self.mean2,
                                  self.sigma, self.alphaL, self. nL,
                                  self.alphaR, self. nL)

        self.frac = ROOT.RooRealVar("frac1", "frac1", frac, 0, 1)
        self.frac.setConstant(True)
        self.pdf = ROOT.RooAddPdf("cb1", "cb1", self.pdf1, self.pdf2,
                                  self.frac)


class Chib2P(object):

    def __init__(self, dm, sigma1, sfrac, frac):
        dm_pdg = pdg.DM1S[2].value()
        self.mean1 = ROOT.RooRealVar("mean_b1_2p", "mean_b1_2p", dm_pdg,
                                    dm_pdg-0.1, dm_pdg+0.1)

        self.sfrac = ROOT.RooRealVar("sfrac2p1p", "sfrac2p1p", sfrac)
        self.sfrac.setConstant(True)

        diff = pdg.DM1S[3].value() - pdg.DM1S[2].value()
        self.dmb2b1 = ROOT.RooRealVar("dmb2b1_2p", "dmb2b1_2p", diff)
        self.dmb2b1.setConstant(True)

        alistb2b1 = ROOT.RooArgList(self.mean1, self.dmb2b1)
        self.mean2 = ROOT.RooFormulaVar("mean_b2_2p", "mean_b2_2p",
                                        "%s+%s" % (self.mean1.GetName(),
                                                   self.dmb2b1.GetName()
                                                   ),
                                        alistb2b1)


        alist = ROOT.RooArgList(self.sfrac, sigma1)
        self.sigma = ROOT.RooFormulaVar("sigma2", "sigma2",
                                        "%s*%s" % (self.sfrac.GetName(),
                                                   sigma1.GetName()), alist)

        self.nR = ROOT.RooRealVar("nr2", "nr2", 5, 1, 5)
        self.alphaR = ROOT.RooRealVar(
            "ar2", "ar2", -0.34, -5, 0)  # -1.23
        self.alphaR.setConstant(True)

        self.nL = ROOT.RooRealVar("nl2", "nl2", 1, 1, 5)
        self.nL.setConstant(True)
        self.alphaL = ROOT.RooRealVar("al2", "al2", 0.8, 0, 5)  # -1.23
        self.alphaL.setConstant(True)

        self.pdf1 = CrystalBallDS("cb1_2", "cb1_2", dm, self.mean1,
                                 self.sigma, self.alphaL, self. nL,
                                 self.alphaR, self. nL)

        self.pdf2 = CrystalBallDS("cb2_2", "cb2_2", dm, self.mean2,
                                 self.sigma, self.alphaL, self. nL,
                                 self.alphaR, self. nL)

        self.frac = ROOT.RooRealVar("frac2", "frac2", frac, 0, 1)
        self.frac.setConstant(True)
        self.pdf = ROOT.RooAddPdf("cb2", "cb2", self.pdf1, self.pdf2,
                                  self.frac)

class Chib3P(object):

    def __init__(self, dm, sigma1, sfrac):
        dm_pdg = pdg.DM1S[4].value()
        self.mean = ROOT.RooRealVar("mean_b1_3p", "mean_b1_3p", dm_pdg,
                                    dm_pdg-0.1, dm_pdg+0.1)

        self.sfrac = ROOT.RooRealVar("sfrac3p1p", "sfrac3p1p", sfrac)
        self.sfrac.setConstant(True)

        alist = ROOT.RooArgList(self.sfrac, sigma1)
        self.sigma = ROOT.RooFormulaVar("sigma3", "sigma3",
                                        "%s*%s" % (self.sfrac.GetName(),
                                                   sigma1.GetName()), alist)

        self.nR = ROOT.RooRealVar("nr3", "nr3", 5, 1, 5)
        self.alphaR = ROOT.RooRealVar(
            "ar3", "ar3", -0.34, -5, 0)  # -1.23
        self.alphaR.setConstant(True)

        self.nL = ROOT.RooRealVar("nl3", "nl3", 1, 1, 5)
        self.nL.setConstant(True)
        self.alphaL = ROOT.RooRealVar("al3", "al3", 0.8, 0, 5)  # -1.23
        self.alphaL.setConstant(True)

        self.pdf = CrystalBallDS("cb3", "cb3", dm, self.mean,
                                 self.sigma, self.alphaL, self. nL,
                                 self.alphaR, self. nL)


class ChibModel(AbstractModel):

    def __init__(self, canvas, dm_begin, dm_end, frac=0.5, nbins=85,
                 bgorder=5):
        super(ChibModel, self).__init__(canvas=canvas, x0=dm_begin,
                                        x1=dm_end, xfield="dm", nbins=nbins)

        self.chib1p = Chib1P(dm=self.x, frac=frac)
        self.chib2p = Chib2P(dm=self.x, sigma1=self.chib1p.sigma, sfrac=1.6,
          frac=frac)
        self.chib3p = Chib3P(dm=self.x, sigma1=self.chib1p.sigma, sfrac=2.1)
        self.bg = Background(dm=self.x, dm_begin=dm_begin, dm_end=dm_end,
                             order=bgorder)

        self.b = ROOT.RooRealVar("B", "Background", 1e+6, 0, 1.e+8)
        self.n1p = ROOT.RooRealVar("N1P", "N1P)", 1e+3, 0, 1.e+7)
        self.n2p = ROOT.RooRealVar("N2P", "N2P)", 1e+2, 0, 1.e+7)
        self.n3p = ROOT.RooRealVar("N3P", "N3P)", 1e+1, 0, 1.e+7)

        self.alist1 = ROOT.RooArgList(
            self.chib1p.pdf,
            self.chib2p.pdf,
            self.chib3p.pdf,
            self.bg.pdf,
        )

        self.alist2 = ROOT.RooArgList(
            self.n1p,
            self.n2p,
            self.n3p,
            self.b
        )

        self.pdf = ROOT.RooAddPdf("model", "model",
                                  self.alist1,
                                  self.alist2)

    def curves(self, frame):
        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.bg.pdf.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kBlue))

        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.chib1p.pdf1.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kMagenta + 1))
        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.chib1p.pdf2.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kGreen + 1))

        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.chib2p.pdf1.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kMagenta + 1))

        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.chib2p.pdf2.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kGreen + 1))

        self.pdf.plotOn(frame,
                        ROOT.RooFit.Components(self.chib3p.pdf.GetName()),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kMagenta + 1))

        # self.pdf.plotOn(frame,
        #                 ROOT.RooFit.Components(self.chib3p.pdf2.GetName()),
        #                 ROOT.RooFit.LineStyle(ROOT.kDashed),
        #                 ROOT.RooFit.LineColor(ROOT.kGreen + 1))