import { useState, useMemo } from 'react';
import type { SourceDoc, ChatFilters } from '../services/chatService';
import SourceCard from './SourceCard';

interface SourcePanelProps {
  sources: SourceDoc[];
  filters?: ChatFilters;
}

export default function SourcePanel({ sources, filters }: SourcePanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const filteredSources = useMemo(() => {
    if (!filters) return sources;
    return sources.filter(doc => {
      if (filters.year_min && doc.year && doc.year < filters.year_min) return false;
      if (filters.year_max && doc.year && doc.year > filters.year_max) return false;
      if (filters.publication_types && filters.publication_types.length > 0) {
        if (!filters.publication_types.includes(doc.publication_type)) return false;
      }
      return true;
    });
  }, [sources, filters]);

  if (!filteredSources.length) return null;

  return (
    <div className="source-panel">
      <button className="source-panel-toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className="source-panel-icon">{isOpen ? '▼' : '▶'}</span>
        <span>Sources ({filteredSources.length} articles)</span>
      </button>
      {isOpen && (
        <div className="source-panel-list">
          {filteredSources.map((doc, i) => (
            <SourceCard key={doc.pmid} doc={doc} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
