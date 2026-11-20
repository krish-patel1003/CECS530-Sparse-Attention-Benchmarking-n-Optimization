// Helpers for building the report with docx-js.
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FONT = "Calibri";
const PAGE_W = 12240, MARGIN = 1080, CONTENT_W = PAGE_W - 2 * MARGIN; // US Letter, 0.75" margins

let figNo = 0, tabNo = 0, shotNo = 0;

function runs(text, base = {}) {
  // **bold**, *italic*, `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text),
  spacing: { after: 60, line: 264 } });
const N = (text) => new Paragraph({ numbering: { reference: "steps", level: 0 }, children: runs(text),
  spacing: { after: 60, line: 264 } });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function Fig(rel, caption, widthIn = 6.9) {
  const shot = /term_/.test(rel);
  const file = path.join(ROOT, rel);
  if (!fs.existsSync(file)) return [P(`[missing figure ${rel}]`)];
  const { w, h } = pngSize(file);
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  const label = shot ? `Screenshot ${++shotNo}` : `Figure ${++figNo}`;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
        altText: { title: label, description: caption, name: label.replace(' ', '') } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `${label}. `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function Tbl(header, rows, widths, caption) {
  const total = widths.reduce((a, b) => a + b, 0);
  const W = widths.map((w) => Math.round((w / total) * CONTENT_W));
  W[W.length - 1] += CONTENT_W - W.reduce((a, b) => a + b, 0);
  const cell = (text, i, head) => new TableCell({
    borders, width: { size: W[i], type: WidthType.DXA },
    shading: head ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(text), { size: 18, bold: head })})],
  });
  tabNo += 1;
  return [
    new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: W,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function Code(lines) {
  return lines.map((l, i) => new Paragraph({
    shading: { fill: "F3F3F3", type: ShadingType.CLEAR, color: "auto" },
    spacing: { after: 0, before: i === 0 ? 80 : 0, line: 240 },
    children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
  })).concat([new Paragraph({ spacing: { after: 100 }, children: [] })]);
}

function csv(name) {
  const file = path.join(ROOT, "results", `${name}.csv`);
  const split = (l) => {               // RFC-4180-ish: handles "quoted, fields"
    const out = []; let cur = "", q = false;
    for (let i = 0; i < l.length; i++) {
      const c = l[i];
      if (q) { if (c === '"' && l[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; }
      else if (c === '"') q = true; else if (c === ",") { out.push(cur); cur = ""; } else cur += c;
    }
    out.push(cur);
    return out;
  };
  const [head, ...lines] = fs.readFileSync(file, "utf8").trim().split("\n");
  const cols = split(head);
  return lines.map((l) => {
    const v = split(l);
    const o = {};
    cols.forEach((c, i) => { const x = v[i]; o[c] = x !== "" && !isNaN(+x) ? +x : x; });
    return o;
  });
}

const styles = {
  default: { document: { run: { font: FONT, size: 21 } } },
  paragraphStyles: [
    { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 40, bold: true, color: "1F3864" },
      paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 } } },
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 30, bold: true, color: "1F3864" },
      paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 25, bold: true, color: "2F5496" },
      paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 22, bold: true, color: "2F5496" },
      paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
  ],
};

const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "steps", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ],
};

module.exports = { P, H1, H2, H3, B, N, Fig, Tbl, Code, csv, runs, styles, numbering, PAGE_W, MARGIN, CONTENT_W };
