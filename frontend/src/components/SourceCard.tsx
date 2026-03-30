import type { SourceDoc } from '../services/chatService';

interface SourceCardProps {
  doc: SourceDoc;
  index: number;
}

export default function SourceCard({ doc, index }: SourceCardProps) {
  const url = `https://pubmed.ncbi.nlm.nih.gov/${doc.pmid}/`;

  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className="source-card">
      <div className="source-card-index">{index + 1}</div>
      <div className="source-card-body">
        <div className="source-card-title">{doc.title || `PMID: ${doc.pmid}`}</div>
        <div className="source-card-meta">
          {doc.first_author && <span>{doc.first_author}</span>}
          {doc.year && <span>{doc.year}</span>}
          {doc.journal && <span>{doc.journal}</span>}
        </div>
        {doc.publication_type && (
          <span className="source-card-badge">{doc.publication_type}</span>
        )}
      </div>
    </a>
  );
}
