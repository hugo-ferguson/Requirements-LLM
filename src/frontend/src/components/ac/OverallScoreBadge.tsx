interface OverallScoreBadgeProps {
  value: number;
}

export function OverallScoreBadge({ value }: OverallScoreBadgeProps) {
  return (
    <span className="rounded-full bg-primary px-3 py-1.5 text-sm font-semibold text-white">
      {value.toFixed(1)}/10
    </span>
  );
}
