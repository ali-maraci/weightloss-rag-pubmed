import { useState } from 'react';
import type { EvidenceTableRow } from '../services/chatService';

interface EvidenceTableProps {
  rows: EvidenceTableRow[];
}

export default function EvidenceTable({ rows }: EvidenceTableProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!rows.length) return null;

  return (
    <div className="evidence-panel">
      <button className="source-panel-toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className="source-panel-icon">{isOpen ? '▼' : '▶'}</span>
        <span>Evidence Table ({rows.length} studies)</span>
      </button>
      {isOpen && (
        <div className="evidence-table-wrapper">
          <table className="evidence-table">
            <thead>
              <tr>
                <th>Study</th>
                <th>Type</th>
                <th>Population</th>
                <th>Intervention</th>
                <th>Comparator</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.pmid}>
                  <td>
                    <a
                      href={`https://pubmed.ncbi.nlm.nih.gov/${row.pmid}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="evidence-study-link"
                    >
                      {row.study}
                    </a>
                  </td>
                  <td><span className="evidence-type-badge">{row.study_type}</span></td>
                  <td>{row.population}</td>
                  <td>{row.intervention}</td>
                  <td>{row.comparator}</td>
                  <td>{row.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
