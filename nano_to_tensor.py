import sys

import awkward as ak
import numpy as np
import torch
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def is_signal_trijet(j1,j2,j3,j4,j5):

    return

input_file = str(sys.argv[1])
print(input_file)
file_list = [s.strip("\n") for s in open(input_file)]
print(file_list)

events = NanoEventsFactory.from_root(
    dict.fromkeys(file_list, "Events"),
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
).events()

cut_events = events[ak.num(events.Jet,axis=1) >= 10]

selected_jets = cut_events.Jet[:,0:10]
trijet = ak.combinations(selected_jets, 5, fields=["j1","j2","j3","j4","j5"])

truth = cut_events.GenPart #[( cut_events.GenPart.hasFlags(["isLastCopy","isPrompt"]))]

parts = ak.drop_none(truth.pdgId,axis=0).compute()
for i in parts:
    print(i)

exit()

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
    },
    with_name="Momentum4D",
)

print(result.mass.compute())

properties = {
    "pt": result.pt.compute(),
    "eta": result.eta.compute(),
    "phi": result.phi.compute(),
    "e": result.e.compute(),
}
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
            trip.append(properties[key][j][i])

        trip.append(mass[j][i])
        event.append(trip)
    events.append(event)

data = np.array(events)

X = np.array([i.flatten() for i in data[:,:,0:-1]])
M = data[:,:,-1:]

X = torch.tensor(X, dtype=torch.float32)
M = torch.tensor(M, dtype=torch.float32)

torch.save(X,"x_tensor.pd")
torch.save(M,"m_tensor.pd")



