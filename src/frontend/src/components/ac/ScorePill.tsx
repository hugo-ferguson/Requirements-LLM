type ScoreLabel = "Relevance" | "Correctness" | "Understandability" | "Coverage";

const TONE_CLASSES: Record<ScoreLabel, string> = {
  Relevance: "bg-blue-100 text-blue-700",
  Correctness: "bg-green-100 text-green-700",
  Understandability: "bg-purple-100 text-purple-700",
  Coverage: "bg-amber-100 text-amber-700",
};

interface ScorePillProps {
  label: ScoreLabel;
  value: number;
}

export function ScorePill({ label, value }: ScorePillProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium ${TONE_CLASSES[label]}`}
    >
      {label} {value.toFixed(1)}
    </span>
  );
}
