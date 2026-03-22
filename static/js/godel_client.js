/**
 * Gödel Client — Browser-side BigInt state & vis-network graph
 *
 * Maintains a local BigInt mirror of the server's Gödel state.
 * Polls /sync to receive O(1) deltas and renders a live knowledge
 * graph using vis-network.
 */
class GodelClient {
    constructor(token = null) {
        this.localState = 1n;
        this.nodes = new vis.DataSet();
        this.edges = new vis.DataSet();
        this.token = token;
    }

    async syncWithServer() {
        try {
            const branchSelector = document.getElementById('branch-selector');
            const branch = branchSelector ? branchSelector.value : "main";
            const headers = { 'Content-Type': 'application/json' };
            if (this.token) {
                headers['Authorization'] = 'Bearer ' + this.token;
            }
            const response = await fetch('/api/v1/quantum/sync', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ client_state_integer: this.localState.toString(), branch: branch })
            });

            if (!response.ok) throw new Error("Sync failed: " + response.status);

            const data = await response.json();

            // Apply delta deletions
            data.delta.delete.forEach(axiom => {
                try { this.edges.remove(axiom); } catch (e) { /* edge may not exist locally */ }
            });

            // Apply delta additions
            data.delta.add.forEach(axiom => {
                const parts = axiom.split("||");
                if (parts.length !== 3) return;
                const [subject, predicate, object_] = parts;

                if (!this.nodes.get(subject)) {
                    this.nodes.add({ id: subject, label: subject });
                }
                if (!this.nodes.get(object_)) {
                    this.nodes.add({ id: object_, label: object_ });
                }

                if (!this.edges.get(axiom)) {
                    this.edges.add({
                        id: axiom,
                        from: subject,
                        to: object_,
                        label: predicate,
                        arrows: 'to'
                    });
                }
            });

            this.localState = BigInt(data.new_global_state);

            // Update HUD
            const stateDisplay = document.getElementById('state-display');
            if (stateDisplay) {
                const s = this.localState.toString();
                stateDisplay.innerText = s.length > 40
                    ? `Gödel State: ${s.substring(0, 40)}… (${s.length} digits)`
                    : `Gödel State: ${s}`;
            }

            const axiomCount = document.getElementById('axiom-count');
            if (axiomCount) {
                axiomCount.innerText = `Axioms: ${this.edges.length}`;
            }

        } catch (error) {
            console.error("Gödel Sync Error:", error);
        }
    }

    startAutoSync(intervalMs = 3000) {
        this.syncWithServer();
        setInterval(() => this.syncWithServer(), intervalMs);
    }
}
