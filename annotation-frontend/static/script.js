// script.js - human-in-the-loop frontend glue
const uploadInput = document.getElementById('videoUpload');
const uploadBtn = document.getElementById('uploadBtn');
const chooseLabel = document.getElementById('chooseLabel');
const fileInfo = document.getElementById('fileInfo');
const statusEl = document.getElementById('status');
const personsGrid = document.getElementById('personsGrid');
const sessionControls = document.getElementById('sessionControls');
const refreshBtn = document.getElementById('refreshBtn');
const confirmBtn = document.getElementById('confirmBtn');
const newPersonBtn = document.getElementById('newPersonBtn');

const modal = document.getElementById('modal');
const modalGrid = document.getElementById('modalGrid');
const modalTitle = document.getElementById('modalTitle');
const closeModal = document.getElementById('closeModal');
const moveToSelect = document.getElementById('moveToSelect');
const moveBtn = document.getElementById('moveBtn');
const deleteBtn = document.getElementById('deleteBtn');

let currentSession = null;
let selected = new Set();

// show file info
uploadInput.addEventListener('change', () => {
  if (!uploadInput.files.length) { fileInfo.textContent = ''; return; }
  const f = uploadInput.files[0];
  fileInfo.textContent = `${f.name} — ${(f.size/1024/1024).toFixed(2)} MB`;
});

// upload & detect
uploadBtn.addEventListener('click', async () => {
  if (!uploadInput.files.length) { alert('Choose a video file first'); return; }
  const fd = new FormData();
  fd.append('file', uploadInput.files[0]);
  statusEl.textContent = 'Uploading and processing...';
  personsGrid.innerHTML = '';
  try {
    const res = await fetch('/upload_video/', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      statusEl.textContent = 'Processing failed';
      console.error(err);
      return;
    }
    const data = await res.json();
    currentSession = data.session_id;
    statusEl.textContent = 'Processing complete — review below.';
    renderPersonsGrid(data.persons);
    sessionControls.classList.remove('hidden');
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Upload error';
  }
});

function renderPersonsGrid(persons) {
  personsGrid.innerHTML = '';
  moveToSelect.innerHTML = '<option value="">Move to...</option>';
  const keys = Object.keys(persons);
  if (keys.length === 0) {
    personsGrid.innerHTML = '<div style="color:#475569">No persons detected</div>';
    return;
  }
  for (const personName of keys) {
    const images = persons[personName];
    const card = document.createElement('div');
    card.className = 'person-card';
    const img = document.createElement('img');
    img.src = images[0];
    img.alt = personName;
    card.appendChild(img);
    const meta = document.createElement('div'); meta.className = 'person-meta';
    const title = document.createElement('div'); title.textContent = personName;
    const count = document.createElement('div'); count.textContent = `${images.length} frames`;
    meta.appendChild(title); meta.appendChild(count);
    card.appendChild(meta);
    card.addEventListener('click', () => openModal(personName, images));
    personsGrid.appendChild(card);

    // add to move options
    const opt = document.createElement('option'); opt.value = personName; opt.textContent = personName;
    moveToSelect.appendChild(opt);
  }
}

// open modal for person
function openModal(personName, images) {
  selected.clear();
  modalTitle.textContent = `${personName} — ${images.length} frames`;
  modalGrid.innerHTML = '';
  for (const src of images) {
    const tile = document.createElement('div'); tile.className = 'tile';
    const img = document.createElement('img'); img.src = src;
    const footer = document.createElement('div'); footer.className = 'tile-footer';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.dataset.src = src;
    checkbox.addEventListener('change', (e) => {
      if (e.target.checked) selected.add(e.target.dataset.src);
      else selected.delete(e.target.dataset.src);
    });
    const small = document.createElement('small'); small.textContent = src.split('/').pop();
    footer.appendChild(checkbox); footer.appendChild(small);
    tile.appendChild(img); tile.appendChild(footer);
    modalGrid.appendChild(tile);
  }
  modal.classList.remove('hidden');
}

// close modal
modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });
closeModal && closeModal.addEventListener('click', () => modal.classList.add('hidden'));

// Move selected
moveBtn && moveBtn.addEventListener('click', async () => {
  const dest = moveToSelect.value;
  if (!dest) return alert('Choose destination person');
  if (selected.size === 0) return alert('Select images to move');
  try {
    for (const src of Array.from(selected)) {
      const resp = await fetch('/move_image/', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ src, dest_person: dest })
      });
      if (!resp.ok) { const err = await resp.json(); alert('Move failed: ' + (err.detail||JSON.stringify(err))); return; }
    }
    await refreshSession();
    modal.classList.add('hidden');
    selected.clear();
  } catch (e) { console.error(e); alert('Move failed'); }
});

// Delete selected
deleteBtn && deleteBtn.addEventListener('click', async () => {
  if (selected.size === 0) return alert('Select images to delete');
  if (!confirm('Delete selected images? This cannot be undone.')) return;
  try {
    for (const src of Array.from(selected)) {
      const resp = await fetch('/delete_image/', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ path: src })
      });
      if (!resp.ok) { const err = await resp.json(); alert('Delete failed: ' + (err.detail||JSON.stringify(err))); return; }
    }
    await refreshSession();
    modal.classList.add('hidden');
    selected.clear();
  } catch (e) { console.error(e); alert('Delete failed'); }
});

// New person
newPersonBtn && newPersonBtn.addEventListener('click', async () => {
  if (!currentSession) return alert('No session open');
  const name = prompt('New person name (optional)') || '';
  const res = await fetch('/add_person/', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ session_id: currentSession, name })
  });
  if (!res.ok) { const err = await res.json(); alert('Create failed'); return; }
  await refreshSession();
});

// Refresh session
refreshBtn && refreshBtn.addEventListener('click', refreshSession);
async function refreshSession() {
  if (!currentSession) return;
  const res = await fetch(`/session/${currentSession}/persons/`);
  if (!res.ok) { statusEl.textContent = 'Session not found'; return; }
  const data = await res.json();
  renderPersonsGrid(data.persons);
}

// Confirm session -> move temp -> backend persons/db
confirmBtn && confirmBtn.addEventListener('click', async () => {
  if (!currentSession) return alert('No session');
  if (!confirm('Confirm & save dataset to database?')) return;
  const res = await fetch('/confirm_session/', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ session_id: currentSession })
  });
  if (!res.ok) { const err = await res.json(); alert('Confirm failed: ' + (err.detail||JSON.stringify(err))); return; }
  const data = await res.json();
  alert('Saved: ' + JSON.stringify(data));
  personsGrid.innerHTML = '';
  sessionControls.classList.add('hidden');
  currentSession = null;
  statusEl.textContent = 'Dataset saved';
});
