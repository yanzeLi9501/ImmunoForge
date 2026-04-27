/* ImmunoForge — Frontend JavaScript */

// ── Pipeline Control ──
async function startPipeline() {
    const species = document.getElementById('species').value;
    const stepsSelect = document.getElementById('steps').value;
    const steps = stepsSelect === 'all' ? null : stepsSelect.split(',');

    const btn = document.getElementById('runPipeline');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    const statusBar = document.getElementById('pipelineStatus');
    statusBar.style.display = 'flex';
    document.getElementById('statusText').textContent = 'Pipeline starting...';

    try {
        const res = await fetch('/api/pipeline/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ species, steps }),
        });
        const data = await res.json();

        if (res.ok) {
            document.getElementById('statusText').textContent =
                `Pipeline running: ${data.steps.join(' → ')}`;
            pollStatus();
        } else {
            document.getElementById('statusText').textContent =
                `Error: ${data.detail || 'Failed to start'}`;
            btn.disabled = false;
            btn.textContent = 'Run Pipeline';
        }
    } catch (err) {
        document.getElementById('statusText').textContent = `Error: ${err.message}`;
        btn.disabled = false;
        btn.textContent = 'Run Pipeline';
    }
}

async function pollStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch('/api/pipeline/status');
            const data = await res.json();

            const statusText = document.getElementById('statusText');
            const btn = document.getElementById('runPipeline');

            if (data.status === 'completed') {
                clearInterval(interval);
                statusText.textContent = 'Pipeline completed!';
                document.querySelector('.spinner').style.display = 'none';
                btn.disabled = false;
                btn.textContent = 'Run Pipeline';
                updateStepCards('completed');
            } else if (data.status === 'failed') {
                clearInterval(interval);
                statusText.textContent = 'Pipeline failed — check logs';
                document.querySelector('.spinner').style.display = 'none';
                btn.disabled = false;
                btn.textContent = 'Run Pipeline';
            } else {
                statusText.textContent = 'Pipeline running...';
            }
        } catch (err) {
            // Server might be busy
        }
    }, 2000);
}

function updateStepCards(status) {
    document.querySelectorAll('.step-card').forEach(card => {
        card.classList.remove('running');
        card.classList.add(status);
    });
}

// ── Quick QC ──
async function runQuickQC() {
    const seq = document.getElementById('qcInput').value.trim().toUpperCase().replace(/[^A-Z]/g, '');
    if (!seq) return;

    const resultDiv = document.getElementById('qcResult');
    resultDiv.style.display = 'block';
    resultDiv.textContent = 'Running QC...';

    try {
        const res = await fetch('/api/qc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sequences: [{ id: 'user_query', sequence: seq }],
            }),
        });
        const data = await res.json();

        const passed = data.passed || [];
        const failed = data.failed || [];
        const entry = passed.length > 0 ? passed[0] : failed[0];

        let html = entry.pass
            ? '✅ PASSED\n'
            : `❌ FAILED: ${(entry.failures || []).join(', ')}\n`;
        html += `\nLength: ${entry.sequence_length} aa`;
        html += `\npI: ${entry.isoelectric_point?.pI || '—'}`;
        html += `\nCys: ${entry.cysteine?.n_cysteines || 0} (${entry.cysteine?.is_even ? 'even' : 'ODD'})`;
        html += `\nAPR: ${entry.aggregation?.apr_count || 0} regions`;
        html += `\nProtease sites: ${JSON.stringify(entry.protease || {})}`;

        resultDiv.textContent = html;
    } catch (err) {
        resultDiv.textContent = `Error: ${err.message}`;
    }
}

// ── Quick Affinity ──
async function runAffinity() {
    const seq = document.getElementById('affinityInput').value.trim().toUpperCase().replace(/[^A-Z]/g, '');
    if (!seq) return;

    const bsa = parseFloat(document.getElementById('bsaInput').value) || 1200;
    const sc = parseFloat(document.getElementById('scInput').value) || 0.65;

    const resultDiv = document.getElementById('affinityResult');
    resultDiv.style.display = 'block';
    resultDiv.textContent = 'Predicting...';

    try {
        const res = await fetch('/api/affinity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: seq, bsa, sc }),
        });
        const data = await res.json();

        const c = data.consensus || {};
        let html = `Consensus K_D: ${c.consensus_kd_nM || '—'} nM (${c.confidence || '—'})\n`;
        html += `\nPRODIGY: ${data.prodigy?.kd_nM || '—'} nM`;
        html += `\nRosetta:  ${data.rosetta?.kd_nM || '—'} nM`;
        html += `\nBSA reg:  ${data.bsa_reg?.kd_nM || '—'} nM`;
        html += `\n\nHot-spot score: ${data.hotspot?.hotspot_score || '—'}`;
        html += `\nHot-spot residues: ${data.hotspot?.n_hotspot_residues || 0}`;

        resultDiv.textContent = html;
    } catch (err) {
        resultDiv.textContent = `Error: ${err.message}`;
    }
}

// ── Quick Codon Opt ──
async function runCodonOpt() {
    const seq = document.getElementById('codonInput').value.trim().toUpperCase().replace(/[^A-Z]/g, '');
    if (!seq) return;

    const species = document.getElementById('codonSpecies').value;
    const system = document.getElementById('codonSystem').value;

    const resultDiv = document.getElementById('codonResult');
    resultDiv.style.display = 'block';
    resultDiv.textContent = 'Optimizing...';

    try {
        const res = await fetch('/api/codon-optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: seq, species, system }),
        });
        const data = await res.json();

        let html = `Species: ${data.species}\n`;
        html += `System: ${data.expression_system}\n`;
        html += `GC content: ${(data.gc_content_cds * 100).toFixed(1)}%\n`;
        html += `T5NT free: ${!data.has_t5nt ? 'Yes ✅' : 'No ❌'}\n`;
        html += `CDS length: ${data.cds_dna?.length || 0} bp\n`;
        html += `Cassette: ${data.cassette?.cassette_length_bp || 0} bp\n`;

        const reSites = data.restriction_sites_found || {};
        const reKeys = Object.keys(reSites);
        if (reKeys.length > 0) {
            html += `\n⚠️ Restriction sites found: ${reKeys.join(', ')}`;
        } else {
            html += '\n✅ No restriction site conflicts';
        }

        html += `\n\nCDS DNA:\n${data.cds_dna || ''}`;

        resultDiv.textContent = html;
    } catch (err) {
        resultDiv.textContent = `Error: ${err.message}`;
    }
}
