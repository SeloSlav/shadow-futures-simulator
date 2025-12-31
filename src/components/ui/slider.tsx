import type { InputHTMLAttributes } from 'react';

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'> {
  value: number[];
  min: number;
  max: number;
  step: number;
  onValueChange: (value: number[]) => void;
}

export function Slider({
  value,
  min,
  max,
  step,
  onValueChange,
  className = '',
  ...props
}: SliderProps) {
  return (
    <input
      type="range"
      className={`slider ${className}`}
      value={value[0]}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onValueChange([parseFloat(e.target.value)])}
      {...props}
    />
  );
}
