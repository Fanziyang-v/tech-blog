const labels = {
  NOTE: 'ⓘ Note',
  TIP: '✦ Tip',
  IMPORTANT: '❖ Important',
  WARNING: '⚠ Warning',
  CAUTION: '⚠ Caution',
};

/** Turn GitHub-style alert markers into styled quotes, preserving Markdown children. */
export default function remarkAlerts() {
  return (tree) => {
    function walk(node) {
      if (node.type === 'blockquote') {
        const paragraph = node.children?.[0];
        const first = paragraph?.children?.[0];
        const match = paragraph?.type === 'paragraph' && first?.type === 'text'
          && first.value.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*(?:\r?\n|$)/);
        if (match) {
          const kind = match[1];
          first.value = first.value.slice(match[0].length);
          if (!first.value) paragraph.children.shift();
          if (!paragraph.children.length) node.children.shift();
          node.data = {
            ...node.data,
            hProperties: { ...node.data?.hProperties, className: ['markdown-alert', `markdown-alert-${kind.toLowerCase()}`] },
          };
          node.children.unshift({
            type: 'paragraph',
            data: { hProperties: { className: ['markdown-alert-title'] } },
            children: [{ type: 'text', value: labels[kind] }],
          });
        }
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}
