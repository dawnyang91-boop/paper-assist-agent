import katex from 'katex';

export function renderMarkdown(markdown: string): string {
  const { text, blocks } = extractDisplayMath(String(markdown || '').replace(/\r\n/g, '\n'));
  const lines = text.split('\n');
  const html: string[] = [];
  let inCode = false;
  let codeLines: string[] = [];
  let listType: 'ul' | 'ol' | null = null;

  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };

  const openList = (type: 'ul' | 'ol') => {
    if (listType === type) return;
    closeList();
    listType = type;
    html.push(`<${type}>`);
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      closeList();
      continue;
    }

    const mathBlock = trimmed.match(/^@@MATH_BLOCK_(\d+)@@$/);
    if (mathBlock) {
      closeList();
      html.push(renderMath(blocks[Number(mathBlock[1])] || '', true));
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    if (unordered) {
      openList('ul');
      html.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      openList('ol');
      html.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${renderInlineMarkdown(line)}</p>`);
  }

  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  }
  closeList();
  return html.join('');
}

function extractDisplayMath(markdown: string): { text: string; blocks: string[] } {
  const blocks: string[] = [];
  const text = markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, formula: string) => {
      const index = blocks.push(formula.trim()) - 1;
      return `\n@@MATH_BLOCK_${index}@@\n`;
    })
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, formula: string) => {
      const index = blocks.push(formula.trim()) - 1;
      return `\n@@MATH_BLOCK_${index}@@\n`;
    });
  return { text, blocks };
}

function renderInlineMarkdown(text: string): string {
  return String(text).split(/(`[^`]*`)/g).map((part) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
    }
    return renderInlineMath(part);
  }).join('');
}

function renderInlineMath(text: string): string {
  const pieces: string[] = [];
  const pattern = /\\\((.+?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    pieces.push(renderPlainInline(text.slice(lastIndex, match.index)));
    pieces.push(renderMath((match[1] || match[2] || '').trim(), false));
    lastIndex = match.index + match[0].length;
  }
  pieces.push(renderPlainInline(text.slice(lastIndex)));
  return pieces.join('');
}

function renderPlainInline(text: string): string {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

function renderMath(formula: string, displayMode: boolean): string {
  if (!formula.trim()) return '';
  try {
    return katex.renderToString(formula, {
      displayMode,
      throwOnError: false,
      strict: false,
      trust: false,
    });
  } catch {
    const escaped = escapeHtml(formula);
    return displayMode
      ? `<pre><code>\\[${escaped}\\]</code></pre>`
      : `<code>\\(${escaped}\\)</code>`;
  }
}

function escapeHtml(value: string): string {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
