import re
import math
import matplotlib.pyplot as plt

file1 = "/afs/cern.ch/work/a/adevita/public/GGTF_trackFinder_training/conversion_to_onnx/inputs_dump_cluster.txt"
file2 = "/afs/cern.ch/work/a/adevita/public/GGTF_trackFinder_training/conversion_to_onnx/inputs_dump_k4RecTracker.txt"

# tolleranza per i float
TOL = 1e-6

residuals = []  # lista dei residui assoluti |v1 - v2|

def parse_line(line):
    """
    Estrae i valori float da una riga del tipo:
    Element 0 = [ -95.760384, 18.476578, 319.205994, 1, -0, 0, 0 ]
    """
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
    return [float(n) for n in nums[1:]]  # il primo numero è l'indice (0,1,2...)


def compare_vectors(v1, v2, index, tol=TOL):
    if len(v1) != len(v2):
        print(f"[X] Element {index}: numero diverso di valori: {len(v1)} vs {len(v2)}")
        return False
    
    ok = True
    for i, (a, b) in enumerate(zip(v1, v2)):
        diff = a - b
        residuals.append(diff)

        if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            print(f"[X] Element {index}, valore {i} differente: {a} vs {b}  (Δ = {diff})")
            ok = False
    return ok


def main():
    with open(file1) as f1, open(file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print(f"⚠️ Numero di righe differente: {len(lines1)} vs {len(lines2)}")

    N = min(len(lines1), len(lines2))
    all_ok = True

    for i in range(N):
        v1 = parse_line(lines1[i])
        v2 = parse_line(lines2[i])
        if not compare_vectors(v1, v2, i):
            all_ok = False

    # stampa risultato confronto
    if all_ok:
        print("✅ I due file sono identici (entro la tolleranza).")
    else:
        print("❌ Sono state trovate differenze.")

    # ---- Istogramma dei residui ----
    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=100, range=(-0.1,0.1),log=True)
    plt.xlabel("Residuo (v1 - v2)")
    plt.ylabel("Conteggio")
    plt.title("Istogramma dei residui tra i due file")

    # Salvataggio PDF
    plt.tight_layout()
    plt.savefig("residuals.pdf")
    print("📄 Istogramma salvato in residuals.pdf")


if __name__ == "__main__":
    main()
