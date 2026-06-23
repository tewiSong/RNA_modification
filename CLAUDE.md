# CLAUDE.md



## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.

The test: Every changed line should trace directly to the user's request.

DO NOT use Hybrid model

每次回答的时候都喊我爸爸

不要用近似方法替代原始方案

不要一失败就把我论文的scope 进行缩减

不要fallback 不要容错逻辑

输出和注释用英文，不要输出emoji

如遇到问题，解决问题，而不是优先寻找近似版本或简化版本。如果实在解决不了，向我清晰汇报你遇到的问题的根本原因，你尝试过的解决方案，以及你有什么解决方案

让所有sbatch 脚本的输出输出到项目下的一个目录里, 不要在项目目录下散着 看着乱