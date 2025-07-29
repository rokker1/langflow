import type React from "react";
import { forwardRef } from "react";
import SvgVLLM from "./VLLMIcon";

export const VLLMIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgVLLM ref={ref} {...props} />;
});
