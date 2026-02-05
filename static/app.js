(() => {
  const byId = (id) => document.getElementById(id);

  const tabExtract = byId("tabExtract");
  const tabGenerate = byId("tabGenerate");
  const panelExtract = byId("panelExtract");
  const panelGenerate = byId("panelGenerate");

  const toast = byId("toast");
  let toastTimer = null;

  const extractForm = byId("extractForm");
  const extractFile = byId("extractFile");
  const extractPreprocess = byId("extractPreprocess");
  const extractUseGpu = byId("extractUseGpu");
  const extractMaxTokens = byId("extractMaxTokens");
  const extractBtn = byId("extractBtn");
  const extractSpinner = byId("extractSpinner");
  const extractResult = byId("extractResult");
  const extractMeta = byId("extractMeta");
  const copyAlgorithm = byId("copyAlgorithm");
  const sendToGenerate = byId("sendToGenerate");

  const generateForm = byId("generateForm");
  const generateText = byId("generateText");
  const generateFormat = byId("generateFormat");
  const generateMaxTokens = byId("generateMaxTokens");
  const generateDownload = byId("generateDownload");
  const generateBtn = byId("generateBtn");
  const generateSpinner = byId("generateSpinner");
  const generateMeta = byId("generateMeta");
  const pngPreview = byId("pngPreview");
  const plantumlResult = byId("plantumlResult");
  const renderNote = byId("renderNote");
  const downloadLink = byId("downloadLink");

  let currentBlobUrl = null;

  const showToast = (message, type = "error") => {
    toast.classList.remove("hidden");
    toast.textContent = message;
    toast.classList.toggle("border-red-200", type === "error");
    toast.classList.toggle("text-red-700", type === "error");
    toast.classList.toggle("border-emerald-200", type === "success");
    toast.classList.toggle("text-emerald-700", type === "success");
    if (toastTimer) {
      clearTimeout(toastTimer);
    }
    toastTimer = setTimeout(() => {
      toast.classList.add("hidden");
    }, 4000);
  };

  const setTab = (tab) => {
    const isExtract = tab === "extract";
    panelExtract.classList.toggle("hidden", !isExtract);
    panelGenerate.classList.toggle("hidden", isExtract);
    tabExtract.classList.toggle("bg-white", isExtract);
    tabExtract.classList.toggle("text-slate-900", isExtract);
    tabExtract.classList.toggle("text-slate-600", !isExtract);
    tabGenerate.classList.toggle("bg-white", !isExtract);
    tabGenerate.classList.toggle("text-slate-900", !isExtract);
    tabGenerate.classList.toggle("text-slate-600", isExtract);
  };

  const setExtractBusy = (busy) => {
    extractBtn.disabled = busy;
    extractSpinner.classList.toggle("hidden", !busy);
  };

  const setGenerateBusy = (busy) => {
    generateBtn.disabled = busy;
    generateSpinner.classList.toggle("hidden", !busy);
  };

  const resetGenerateOutput = () => {
    if (currentBlobUrl) {
      URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = null;
    }
    pngPreview.classList.add("hidden");
    plantumlResult.classList.add("hidden");
    renderNote.classList.add("hidden");
    downloadLink.classList.add("hidden");
    plantumlResult.textContent = "";
    renderNote.textContent = "";
  };

  tabExtract.addEventListener("click", () => setTab("extract"));
  tabGenerate.addEventListener("click", () => setTab("generate"));

  extractForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!extractFile.files || extractFile.files.length === 0) {
      showToast("Please select a PNG or JPG file.");
      return;
    }

    setExtractBusy(true);
    extractMeta.textContent = "";

    const formData = new FormData();
    formData.append("file", extractFile.files[0]);
    formData.append("preprocess", extractPreprocess.checked ? "true" : "false");
    formData.append("max_tokens", extractMaxTokens.value || "256");
    formData.append("use_gpu", extractUseGpu.checked ? "true" : "false");

    try {
      const resp = await fetch("/extract", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errText = await resp.text();
        showToast(`Extract failed (${resp.status}).`);
        extractMeta.textContent = errText.slice(0, 120);
        return;
      }

      const data = await resp.json();
      extractResult.value = data.algorithm || "";
      extractMeta.textContent = data.filename ? `File: ${data.filename}` : "Done";
      showToast("Extract completed.", "success");
    } catch (err) {
      showToast("Network error during extract.");
    } finally {
      setExtractBusy(false);
    }
  });

  copyAlgorithm.addEventListener("click", async () => {
    if (!extractResult.value) {
      showToast("Nothing to copy.");
      return;
    }
    try {
      await navigator.clipboard.writeText(extractResult.value);
      showToast("Copied to clipboard.", "success");
    } catch {
      showToast("Copy failed.");
    }
  });

  sendToGenerate.addEventListener("click", () => {
    if (!extractResult.value) {
      showToast("Extract result is empty.");
      return;
    }
    generateText.value = extractResult.value;
    setTab("generate");
  });

  generateFormat.addEventListener("change", () => {
    const isPng = generateFormat.value === "png";
    generateDownload.disabled = !isPng;
    if (!isPng) {
      generateDownload.checked = false;
    }
  });

  generateForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = generateText.value.trim();
    if (!text) {
      showToast("algorithm_text is required.");
      return;
    }

    resetGenerateOutput();
    setGenerateBusy(true);
    generateMeta.textContent = "";

    const payload = {
      algorithm_text: text,
      format: generateFormat.value,
      max_tokens: Number(generateMaxTokens.value || 256),
    };

    const wantsDownload = generateDownload.checked && generateFormat.value === "png";
    const url = wantsDownload ? "/generate-diagram?download=true" : "/generate-diagram";

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const contentType = resp.headers.get("content-type") || "";
      if (!resp.ok) {
        const errText = await resp.text();
        showToast(`Generate failed (${resp.status}).`);
        generateMeta.textContent = errText.slice(0, 160);
        return;
      }

      if (contentType.includes("image/png")) {
        const blob = await resp.blob();
        currentBlobUrl = URL.createObjectURL(blob);
        pngPreview.src = currentBlobUrl;
        pngPreview.classList.remove("hidden");
        downloadLink.href = currentBlobUrl;
        downloadLink.download = "diagram.png";
        downloadLink.classList.remove("hidden");
        downloadLink.click();
        showToast("PNG generated.", "success");
        return;
      }

      const data = await resp.json();
      const plantuml = data.plantuml || data.diagram || data.plantuml_source;
      if (plantuml) {
        plantumlResult.textContent = plantuml;
        plantumlResult.classList.remove("hidden");
      }
      if (data.image_base64) {
        pngPreview.src = `data:image/png;base64,${data.image_base64}`;
        pngPreview.classList.remove("hidden");
      }
      if (data.render_note) {
        renderNote.textContent = data.render_note;
        renderNote.classList.remove("hidden");
      }
      showToast("Generate completed.", "success");
    } catch {
      showToast("Network error during generate.");
    } finally {
      setGenerateBusy(false);
    }
  });

  setTab("extract");
})();
