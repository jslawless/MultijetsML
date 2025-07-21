import sys

import awkward as ak
import numpy as np
import torch
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def tight_jets(ev, jet_eta_cut: float = 2.4, jet_pt_cut: float = 30):
    jet_cut = (
        (abs(ev.Jet.eta) < jet_eta_cut)
        & (ev.Jet.pt > jet_pt_cut)
        & (ev.Jet.neHEF < 0.90)
        & (ev.Jet.neEmEF < 0.90)
        & (ev.Jet.nConstituents > 1)
        & (ev.Jet.muEF < 0.80)
        & (ev.Jet.chHEF > 0.01)
        # & (ev.Jet.nCh > 0) \
        & (ev.Jet.chEmEF < 0.80)
    )

    tj = ev.Jet[jet_cut]
    ev["GenPart","signal"] = 0
    gp = ev.GenPart[(ev.GenPart.hasFlags(["isLastCopy","isPrompt"]))]

    slg  = gp[gp.pdgId == 1000021]

    ch1 = slg[:,0].distinctChildren
    ch1["signal"] = 1
    nch1 = ak.flatten(ch1.distinctChildren,axis=2)
    nch1["signal"] = 1
    nnch1 = ak.flatten(nch1.distinctChildren,axis=2)
    nnch1["signal"] = 1
    nnnch1 = ak.flatten(nnch1.distinctChildren,axis=2)
    nnnch1["signal"] = 1
    nnnnch1 = ak.flatten(nnnch1.distinctChildren,axis=2)
    nnnnch1["signal"] = 1
    nnnnnch1 = ak.flatten(nnnnch1.distinctChildren,axis=2)
    nnnnnch1["signal"] = 1

    ch2 = slg[:,1].distinctChildren
    ch2["signal"] = 2
    nch2 = ak.flatten(ch2.distinctChildren,axis=2)
    nch2["signal"] = 2
    nnch2 = ak.flatten(nch2.distinctChildren,axis=2)
    nnch2["signal"] = 2
    nnnch2 = ak.flatten(nnch2.distinctChildren,axis=2)
    nnnch2["signal"] = 2
    nnnnch2 = ak.flatten(nnnch2.distinctChildren,axis=2)
    nnnnch2["signal"] = 2
    nnnnnch2 = ak.flatten(nnnnch2.distinctChildren,axis=2)
    nnnnnch2["signal"] = 2
    chs = ak.concatenate([ch1, nch1, nnch1,nnnch1,nnnnch1,nnnnnch1, ch2, nch2, nnch2, nnnch2,nnnnch2,nnnnnch2], axis=1)
    ev["SlimGenPart"] = gp
    tj["MatchedGenPart"] = tj.nearest(chs,threshold=0.4)
    tj["signal"] = tj.MatchedGenPart.signal
    myslice = (tj.signal == 1) | (tj.signal == 2)
    ev["TightJet"] = tj[ak.fill_none(myslice,False)]


def is_signal_trijet(j1,j2,j3,j4,j5):

    return j1.signal == j2.signal == j3.signal == j4.signal == j5.signal

input_file = str(sys.argv[1])
print(input_file)
file_list = [s.strip("\n") for s in open(input_file)]
print(file_list)

events = NanoEventsFactory.from_root(
    dict.fromkeys(file_list, "Events"),
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
).events()

# print("Number of events: ", ak.num(events,axis=0).compute())

tight_jets(events)
cut_events = events[ak.num(events.Jet,axis=1) >= 10]
# print("Number of events: ", ak.num(cut_events,axis=0).compute())
selected_jets = cut_events.Jet[:,0:10]
trijet = ak.combinations(selected_jets, 5, fields=["j1","j2","j3","j4","j5"])


result = ak.zip(
    {
        "j1": trijet.j1,
        "j2": trijet.j2,
        "j3": trijet.j3,
        "j4": trijet.j4,
        "j5": trijet.j5,
        "px": trijet.j1.px + trijet.j2.px + trijet.j3.px + trijet.j4.px + trijet.j5.px,
        "py": trijet.j1.py + trijet.j2.py + trijet.j3.py + trijet.j4.py + trijet.j5.py,
        "pz": trijet.j1.pz + trijet.j2.pz + trijet.j3.pz + trijet.j4.pz + trijet.j5.pz,
        "e": trijet.j1.E + trijet.j2.E + trijet.j3.E + trijet.j4.E + trijet.j5.E,
        "match": is_signal_trijet(trijet.j1, trijet.j2, trijet.j3, trijet.j4, trijet.j5),
    },
    with_name="Momentum4D",
)

properties = {
    "pt": result.pt.compute(),
    "eta": result.eta.compute(),
    "phi": result.phi.compute(),
    "e": result.e.compute(),
}
match = result.match.compute()
mass = result.mass.compute()

events = []
print(len(mass))
for j in range(len(mass)):
    if j % 500 == 0:
        print(str(j) + "...")
    event = []
    for i in range(252): # change to pairs instead of triplets
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



