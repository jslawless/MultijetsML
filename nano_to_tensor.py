import sys

import awkward as ak
import numpy as np
import torch
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

print("loaded libraries")

class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,
        "ScoutingPFJet": "Jet",
        "ScoutingJet": "Jet",
        "ScoutingFatJet": "Jet",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingPrimaryVertex":"Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon",
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon",

    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references,
    }

def tight_jets(ev, jet_eta_cut: float = 2.4, jet_pt_cut: float = 30):
    jet_cut = (
        (abs(ev.ScoutingPFJet.eta) < jet_eta_cut)
        & (ev.ScoutingPFJet.pt > jet_pt_cut)
        # & (ev.ScoutingPFJet.neHEF < 0.90)
        # & (ev.ScoutingPFJet.neEmEF < 0.90)
        # & (ev.ScoutingPFJet.nConstituents > 1)
        # & (ev.ScoutingPFJet.muEF < 0.80)
        # & (ev.ScoutingPFJet.chHEF > 0.01)
        # & (ev.ScoutingPFJet.nCh > 0) \
        # & (ev.ScoutingPFJet.chEmEF < 0.80)
    )

    # gp = ak.Array({"genPartIdxMother": dak.Scalar(-999,name="genPartIdxMother")})


    # gp.pdgId = -999
    # gp.genPartIdxMother = -1

    tj = ev.ScoutingPFJet[jet_cut]
    ev["GenPart","signal"] = 0
    gp = ev.GenPart[(ev.GenPart.hasFlags(["isLastCopy"]))]

    b_one = gp[gp.pdgId == 5]
    b_one["signal"] = 1
    w_one = gp[gp.pdgId == 24].distinctChildren
    j_one = ak.flatten(w_one[w_one.hasFlags(["isLastCopy"])],axis=2)
    j_one["signal"] = 1

    b_two = gp[gp.pdgId == -5]
    b_two["signal"] = 2
    w_two = gp[gp.pdgId == -24].distinctChildren
    j_two = ak.flatten(w_two[w_two.hasFlags(["isLastCopy"])],axis=2)
    j_two["signal"] = 2

    chs = ak.concatenate([b_one, j_one, b_two, j_two], axis=1)
    ev["SlimGenPart"] = gp

    tj["MatchedGenPart"] = tj.nearest(chs,threshold=0.4)

    tj["OGGenPart"] = tj.nearest(gp,threshold=0.4)
    tj["signal"] = tj.MatchedGenPart.signal
    myslice = (tj.signal == 1) | (tj.signal == 2)
    ev["TightJet"] = tj[ak.fill_none(myslice,False)]


def is_signal_trijet(j1,j2,j3):
    return (j1.signal == j2.signal) & (j2.signal == j3.signal)

input_file = str(sys.argv[1])
print(input_file)
# file_list = [s.strip("\n") for s in open(input_file)]
# print(file_list)


print(dict.fromkeys([input_file], "Events"))
events = NanoEventsFactory.from_root(
    dict.fromkeys([input_file], "Events"),
    schemaclass=ScoutingNanoAODSchema,
    metadata={"dataset": "DYJets"},
).events()

print("loaded events")
# print("Number of events: ", ak.num(events,axis=0).compute())

tight_jets(events)
cut_events = events[ak.num(events.TightJet,axis=1) >= 6]

# print("Number of events: ", ak.num(cut_events,axis=0).compute())
selected_jets = cut_events.TightJet[:,0:6]
trijet = ak.combinations(selected_jets, 3, fields=["j1","j2","j3"])


result = ak.zip(
    {
        "j1": trijet.j1,
        "j2": trijet.j2,
        "j3": trijet.j3,
        "px": trijet.j1.px + trijet.j2.px + trijet.j3.px,
        "py": trijet.j1.py + trijet.j2.py + trijet.j3.py,
        "pz": trijet.j1.pz + trijet.j2.pz + trijet.j3.pz,
        "e": trijet.j1.E + trijet.j2.E + trijet.j3.E,
        "match": is_signal_trijet(trijet.j1, trijet.j2, trijet.j3),
    },
    with_name="Momentum4D",
)

print("formed trijets")

properties = {
    "pt": result.pt.compute(),
    "eta": result.eta.compute(),
    "phi": result.phi.compute(),
    "e": result.e.compute(),
}
print("computed trijets 1")
match = result.match.compute()
mass = result.mass.compute()
print("computed trijets 2")
events = []
print(len(mass))
for j in range(len(mass)):
    if j % 500 == 0:
        print(str(j) + "...")
    event = []
    for i in range(20): # change to pairs instead of triplets
        trip = []
        for key in properties:
            print(i)
            print(j)
            print(key)
            print(properties[key][j])
            trip.append(properties[key][j][i])

        trip.append(match[j][i])
        trip.append(mass[j][i])
        event.append(trip)
    events.append(event)

data = np.array(events)

X = np.array([i.flatten() for i in data[:,:,0:-2]])
Y = data[:,:,-2:-1]
M = data[:,:,-1:]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
M = torch.tensor(M, dtype=torch.float32)

torch.save(X,"x_tensor.pd")
torch.save(Y,"y_tensor.pd")
torch.save(M,"m_tensor.pd")