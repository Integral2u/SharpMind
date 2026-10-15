using System.Text;
using System.Text.RegularExpressions;
using SharpMind.Core;
using SharpMind.Core.Diagnostics;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Executes the Jinja2 subset used by GGUF <c>tokenizer.chat_template</c> strings.
///
/// Supported constructs (covering every template seen in practice):
///   {%- for message in messages %} … {% endfor %}
///   {%- if expr %} … {%- elif expr %} … {%- else %} … {%- endif %}
///   {%- set var = expr %}
///   {%- set ns = namespace(field=value, …) %}  and  ns.field  assignments
///   {{ expr }}          — output expression
///   loop.first / loop.last / loop.index0
///   message['role'] / message.role / message['content'] / message.content
///   messages[0]['role'] / messages[loop.index0 ± N]
///   bos_token, eos_token  — resolved from the Tokenizer at render time
///   add_generation_prompt — bool passed by the caller (always true for inference)
///   X is none / X is not none / X is defined / not X is defined
///   'str' in X  (substring / key test)
///   X + Y  (string concatenation)
///   content.split('delim')[-1]  (last segment after split)
///   X | trim  (whitespace trim filter)
///   tool_calls / tool call blocks are intentionally skipped (plain-text chat only)
/// </summary>
public sealed class JinjaTemplateFormatter(string template) : IChatPromptFormatter
{
    private readonly string _template = template;

    /// <summary>Errors accumulated during the last <see cref="Format"/> call.</summary>
    public IReadOnlyList<string>? LastErrors { get; private set; }

    /// <summary>
    /// Jinja templates embed the model's own end-of-turn token, so the only
    /// reliable terminator is EOS (handled via StopTokenIds) — no extra plain-text
    /// stop strings are defined.
    /// </summary>
    public IReadOnlyList<string> DefaultStopStrings => [];

    // Public entry point

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        // Build the variable context that templates can reference.
        string bosToken = addBos && tokenizer.BosId >= 0
            ? tokenizer.IdToToken(tokenizer.BosId) : "";
        string eosToken = tokenizer.EosId >= 0
            ? tokenizer.IdToToken(tokenizer.EosId) : "";

        // Convert ChatMessage list to the dict-of-strings representation
        // that templates expect: message['role'] and message['content'].
        var messages = history.Select(m => new JinjaDict
        {
            ["role"] = RoleName(m.Role),
            ["content"] = (object)m.Content
        }).ToList();

        // Convert tool definitions JSON to JinjaDict list for template access.
        var toolsList = JsonToJinjaDictList(toolsJson);

        var env = new JinjaEnv();
        env.Set("messages", messages);
        env.Set("bos_token", bosToken);
        env.Set("eos_token", eosToken);
        env.Set("add_generation_prompt", (object)true);  // always true for inference
        env.Set("enable_thinking", (object)enableThinking);
        env.Set("preserve_thinking", (object)false);
        // Always expose tools/custom_tools (empty when none) so templates that
        // test them see a falsy value rather than an undefined-variable error.
        env.Set("tools", (object)toolsList);
        env.Set("custom_tools", (object)toolsList);
        env.Set("tools_in_user_message", (object)false);
        env.Set("date_string", (object)DateTime.Now.ToString("dd MMM yyyy"));
        env.Set("strftime_now", (object)DateTime.Now.ToString("dd MMM yyyy"));

        var errors = new List<string>();
        var tokens = Tokenise(_template);
        var ctx = new Ctx(tokens, errors);
        RegisterMacros(tokens, 0, tokens.Count, ctx);
        var sb = new StringBuilder();
        Execute(tokens, 0, tokens.Count, env, sb, ctx);
        LastErrors = errors.Count > 0 ? errors : null;
        return sb.ToString();
    }

    /// <summary>
    /// Per-render state shared by the token walker and the expression evaluator:
    /// the flattened token list (macro bodies are re-entered here) and the macro
    /// table. Threading it explicitly keeps <see cref="Eval(string, JinjaEnv, List{string}?)"/>
    /// usable on its own (tests call it directly) with an empty context.
    /// </summary>
    private sealed class Ctx(List<Token> tokens, List<string> errors)
    {
        public List<Token> Tokens { get; } = tokens;
        public List<string> Errors { get; } = errors;
        public Dictionary<string, MacroDef> Macros { get; } = new(StringComparer.Ordinal);
        public JinjaEnv? Root { get; set; }
    }

    private sealed record MacroParam(string Name, string? DefaultExpr);

    private sealed record MacroDef(string Name, IReadOnlyList<MacroParam> Params, int BodyStart, int BodyEnd);

    /// <summary>
    /// Pre-scan for <c>{% macro name(params) %} … {% endmacro %}</c> and record each
    /// body's token span so a call can re-enter it with bound arguments. Macros do
    /// not nest in the templates seen in practice, but the span is found with a
    /// depth counter so a nested definition cannot silently swallow the outer body.
    /// </summary>
    private static void RegisterMacros(List<Token> tokens, int start, int end, Ctx ctx)
    {
        int i = start;
        while (i < end)
        {
            var tok = tokens[i];
            if (tok.Kind == TKind.Tag && tok.Body.StartsWith("macro ", StringComparison.Ordinal))
            {
                var m = RegexGenerated.JinjaMacroDef.Match(tok.Body);
                if (m.Success)
                {
                    int bodyEnd = FindMatchingEnd(tokens, i, end, "macro", "endmacro");
                    var ps = new List<MacroParam>();
                    string paramStr = m.Groups[2].Value;
                    foreach (string raw in SplitTopLevel(paramStr, ','))
                    {
                        string p = raw.Trim();
                        if (p.Length == 0) continue;
                        int eq2 = p.IndexOf('=');
                        if (eq2 > 0)
                            ps.Add(new MacroParam(p[..eq2].Trim(), p[(eq2 + 1)..].Trim()));
                        else
                            ps.Add(new MacroParam(p, null));
                    }
                    ctx.Macros[m.Groups[1].Value] = new MacroDef(m.Groups[1].Value, ps, i + 1, bodyEnd);
                }
                i = (m.Success ? FindMatchingEnd(tokens, i, end, "macro", "endmacro") : i) + 1;
                continue;
            }
            i++;
        }
    }

    // Tokeniser
    // Splits the template into a flat list of Token objects.

    public enum TKind { Text, Output, Tag }

    public sealed record Token(TKind Kind, string Body, bool StripLeft, bool StripRight);

    private static List<Token> Tokenise(string src)
    {
        var result = new List<Token>();
        int pos = 0;
        while (pos < src.Length)
        {
            int tag = src.IndexOf("{%", pos, StringComparison.Ordinal);
            int expr = src.IndexOf("{{", pos, StringComparison.Ordinal);
            int comment = src.IndexOf("{#", pos, StringComparison.Ordinal);

            // Find whichever opener comes first.
            int next = (tag < 0 && expr < 0 && comment < 0) ? -1
                      : EarliestNonNegative(tag, expr, comment);

            if (next < 0)
            {
                if (pos < src.Length) result.Add(new Token(TKind.Text, src[pos..], false, false));
                break;
            }

            if (next > pos)
                result.Add(new Token(TKind.Text, src[pos..next], false, false));

            if (next == comment)
            {
                // {# comment #}
                int end = src.IndexOf("#}", next + 2, StringComparison.Ordinal);
                if (end < 0) { result.Add(new Token(TKind.Text, src[next..], false, false)); break; }
                string raw = src[(next + 2)..end];
                bool sr = raw.EndsWith('-');
                ApplyStripLeftPrevText(result, raw.StartsWith('-'));
                pos = end + 2;
                if (sr) ConsumeWs(ref pos, src);
            }
            else if (next == expr && (tag < 0 || expr < tag))
            {
                int end = src.IndexOf("}}", next + 2, StringComparison.Ordinal);
                if (end < 0) { result.Add(new Token(TKind.Text, src[next..], false, false)); break; }
                string raw = src[(next + 2)..(end)];
                bool sr = raw.EndsWith('-');
                ApplyStripLeftPrevText(result, raw.StartsWith('-'));
                string body = raw.TrimStart('-').TrimEnd('-').Trim();
                result.Add(new Token(TKind.Output, body, false, sr));
                pos = end + 2;
                if (sr) ConsumeWs(ref pos, src);
            }
            else
            {
                int end = src.IndexOf("%}", next + 2, StringComparison.Ordinal);
                if (end < 0) { result.Add(new Token(TKind.Text, src[next..], false, false)); break; }
                string raw = src[(next + 2)..end];
                bool sr = raw.EndsWith('-');
                ApplyStripLeftPrevText(result, raw.StartsWith('-'));
                string body = raw.Trim().TrimStart('-').TrimEnd('-').Trim();
                result.Add(new Token(TKind.Tag, body, false, sr));
                pos = end + 2;
                if (sr) ConsumeWs(ref pos, src);
            }
        }
        return result;
    }

    private static void ConsumeWs(ref int pos, string src)
    {
        while (pos < src.Length && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r'))
            pos++;
    }

    private static void ApplyStripLeftPrevText(List<Token> result, bool stripLeft)
    {
        if (!stripLeft) return;
        while (result.Count > 0 && result[^1] is Token { Kind: TKind.Text } t)
        {
            string trimmed = t.Body.TrimEnd();
            if (trimmed.Length == t.Body.Length) break; // no trailing whitespace
            if (trimmed.Length == 0)
                result.RemoveAt(result.Count - 1);
            else
                result[^1] = t with { Body = trimmed };
            if (trimmed.Length > 0) break;
        }
    }

    private static int EarliestNonNegative(int a, int b, int c)
    {
        int min = int.MaxValue;
        if (a >= 0 && a < min) min = a;
        if (b >= 0 && b < min) min = b;
        if (c >= 0 && c < min) min = c;
        return min == int.MaxValue ? -1 : min;
    }

    // Executor

    private static void Execute(
        List<Token> tokens, int start, int end,
        JinjaEnv env, StringBuilder sb, Ctx ctx)
    {
        int i = start;
        while (i < end)
        {
            var tok = tokens[i];

            // strip-left: trim trailing whitespace from output so far
            if (tok.StripLeft)
                while (sb.Length > 0 && char.IsWhiteSpace(sb[^1]))
                    sb.Length--;

            if (tok.Kind == TKind.Text)
            {
                string text = tok.Body;
                // strip-right from previous output already handled in Tokenise (newline skip)
                sb.Append(text);
                i++;
                continue;
            }

            if (tok.Kind == TKind.Output)
            {
                var val = Eval(tok.Body, env, ctx);
                if (val != null) sb.Append(Stringify(val));
                i++;
                continue;
            }

            // Tag
            string tag = tok.Body;

            if (tag.StartsWith("for ", StringComparison.Ordinal))
            {
                i = ExecuteFor(tokens, i, end, env, sb, ctx);
                continue;
            }

            if (tag.StartsWith("if ", StringComparison.Ordinal) || tag == "if")
            {
                i = ExecuteIf(tokens, i, end, env, sb, ctx);
                continue;
            }

            // Macro definitions are pre-registered; skip the body entirely so it
            // does not execute inline at its definition site.
            if (tag.StartsWith("macro ", StringComparison.Ordinal))
            {
                i = FindMatchingEnd(tokens, i, end, "macro", "endmacro") + 1;
                continue;
            }

            if (tag.StartsWith("set ", StringComparison.Ordinal))
            {
                i = ExecuteSet(tokens, i, end, env, sb, ctx);
                continue;
            }

            // Unrecognized tag — log error and skip
            ctx.Errors.Add($"Unrecognized Jinja tag at token {i}: '{tag}'");
            i++;
        }
    }

    // For loop

    private static int ExecuteFor(
        List<Token> tokens, int forIdx, int end,
        JinjaEnv env, StringBuilder sb, Ctx ctx)
    {
        // Parse "for VAR[, VAR...] in EXPR"
        var m = RegexGenerated.JinjaForVarInExpr.Match(tokens[forIdx].Body);
        if (!m.Success)
        {
            ctx.Errors.Add($"JINJA ERROR: failed to parse 'for' tag at token {forIdx}: '{tokens[forIdx].Body}'");
            return forIdx + 1;
        }

        string[] varNames = m.Groups[1].Value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        string iterExpr = m.Groups[2].Value.Trim();
        var iterable = Eval(iterExpr, env, ctx);

        // Find matching endfor
        int bodyStart = forIdx + 1;
        int bodyEnd = FindMatchingEnd(tokens, forIdx, end, "for", "endfor");

        var items = AsIterableList(iterable);
        if (items is not null)
        {
            int count = items.Count;
            for (int idx = 0; idx < count; idx++)
            {
                var child = env.Push();
                object? item = items[idx];
                if (varNames.Length > 1 && item is List<object?> tuple)
                {
                    for (int v = 0; v < varNames.Length; v++)
                        child.Set(varNames[v], v < tuple.Count ? tuple[v] : null);
                }
                else
                {
                    child.Set(varNames[0], item);
                }

                var loop = new JinjaDict
                {
                    ["first"] = (object)(idx == 0),
                    ["last"] = (object)(idx == count - 1),
                    ["index0"] = (object)(long)idx,
                    ["index"] = (object)(long)(idx + 1)
                };
                child.Set("loop", loop);

                Execute(tokens, bodyStart, bodyEnd, child, sb, ctx);
            }
        }
        else if (iterable is not null)
        {
            ctx.Errors.Add($"JINJA ERROR: 'for {m.Groups[1].Value} in {iterExpr}' — expression is not iterable (got {iterable.GetType().Name})");
        }

        return bodyEnd + 1; // +1 to skip the endfor token
    }

    /// <summary>
    /// Normalises any iterable value the engine can hold into a
    /// <c>List&lt;object?&gt;</c> so a single <c>for</c> path handles messages,
    /// split() results, ranges and dictsort pairs alike.
    /// </summary>
    private static List<object?>? AsIterableList(object? v) => v switch
    {
        List<object?> l => l,
        List<JinjaDict> l => l.Cast<object?>().ToList(),
        JinjaDict d => d.Cast<object?>().ToList(),
        _ => null
    };

    // If / elif / else

    private static int ExecuteIf(
        List<Token> tokens, int ifIdx, int end,
        JinjaEnv env, StringBuilder sb, Ctx ctx)
    {
        // Collect all branches: [(condition_expr, start, end), …]
        // null condition = else branch
        var branches = new List<(string? Cond, int Start, int End)>();
        int depth = 0;
        int branchStart = ifIdx + 1;
        string? curCond = tokens[ifIdx].Body["if ".Length..].Trim();

        for (int i = ifIdx + 1; i < end; i++)
        {
            if (tokens[i].Kind != TKind.Tag) continue;
            string t = tokens[i].Body;

            if (t.StartsWith("for ", StringComparison.Ordinal) ||
                t.StartsWith("if ", StringComparison.Ordinal)) { depth++; continue; }

            if ((t == "endfor" || t == "endif") && depth > 0) { depth--; continue; }

            if (depth > 0) continue;

            if (t == "endif")
            {
                branches.Add((curCond, branchStart, i));
                foreach (var (cond, bs, be) in branches)
                {
                    if (cond == null || IsTruthy(Eval(cond, env, ctx)))
                    {
                        Execute(tokens, bs, be, env, sb, ctx);
                        break;
                    }
                }
                return i + 1;
            }

            if (t.StartsWith("elif ", StringComparison.Ordinal) ||
                t.StartsWith("else if ", StringComparison.Ordinal))
            {
                branches.Add((curCond, branchStart, i));
                curCond = t.StartsWith("elif ") ? t["elif ".Length..].Trim()
                                                       : t["else if ".Length..].Trim();
                branchStart = i + 1;
                continue;
            }

            if (t == "else")
            {
                branches.Add((curCond, branchStart, i));
                curCond = null;
                branchStart = i + 1;
            }
        }

        // Fell off the end without finding endif — execute what we have
        branches.Add((curCond, branchStart, end));
        foreach (var (cond, bs, be) in branches)
        {
            if (cond == null || IsTruthy(Eval(cond, env, ctx)))
            {
                Execute(tokens, bs, be, env, sb, ctx);
                break;
            }
        }
        return end;
    }

    // Set

    private static int ExecuteSet(List<Token> tokens, int tagIdx, int end, JinjaEnv env, StringBuilder sb, Ctx ctx)
    {
        string tag = tokens[tagIdx].Body;
        // set ns = namespace(field=val, …)
        var nsMatch = RegexGenerated.JinjaSetNsFieldValue.Match(tag);// Regex.Match(tag,@"^set\s+(\w+)\s*=\s*namespace\s*\((.*)?\)\s*$",RegexOptions.Singleline);
        if (nsMatch.Success)
        {
            string name = nsMatch.Groups[1].Value;
            string args = nsMatch.Groups[2].Value;
            var ns = new JinjaNamespace();
            foreach (Match kv in RegexGenerated.JinjaNamespace.Matches(args)) // Regex.Matches(args, @"(\w+)\s*=\s*([^,]+)"))
            {
                string k = kv.Groups[1].Value.Trim();
                string v = kv.Groups[2].Value.Trim();
                    ns.Set(k, Eval(v, env, ctx) ?? (object)"");
            }
            env.Set(name, ns);
            return tagIdx + 1;
        }

        // set ns.field = expr
        var dotMatch = RegexGenerated.JinjaNamespaceDotFieldEqExpr.Match(tag);// Regex.Match(tag, @"^set\s+(\w+)\.(\w+)\s*=\s*(.+)$");
        if (dotMatch.Success)
        {
            string obj = dotMatch.Groups[1].Value;
            string field = dotMatch.Groups[2].Value;
            string expr = dotMatch.Groups[3].Value.Trim();
            var target = env.Get(obj);
            if (target is JinjaNamespace jns)
                jns.Set(field, Eval(expr, env, ctx) ?? (object)"");
            else if (target is JinjaDict jd)
                jd[field] = Eval(expr, env, ctx) ?? (object)"";
            else
                ctx.Errors.Add($"JINJA ERROR: 'set {obj}.{field} = {expr}' — '{obj}' is not a namespace or dict (got {target?.GetType().Name ?? "null"})");
            return tagIdx + 1;
        }

        // set var = expr
        var simple = RegexGenerated.JinjaSetVarEqExpr.Match(tag);// Regex.Match(tag, @"^set\s+(\w+)\s*=\s*(.+)$");
        if (simple.Success)
        {
            string name = simple.Groups[1].Value;
            string expr = simple.Groups[2].Value.Trim();
            env.Set(name, Eval(expr, env, ctx) ?? (object)"");
            return tagIdx + 1;
        }

        // Capture block: set name (no =) — executed between tag and endset
        var capM = Regex.Match(tag, @"^set\s+(\w+)\s*$", RegexOptions.Singleline);
        if (capM.Success)
        {
            string name = capM.Groups[1].Value;
            int depth = 0;
            int bodyStart = tagIdx + 1;
            for (int i = tagIdx + 1; i < end; i++)
            {
                if (tokens[i].Kind != TKind.Tag) continue;
                string t = tokens[i].Body;
                if (t.StartsWith("for ", StringComparison.Ordinal) ||
                    t.StartsWith("if ", StringComparison.Ordinal) ||
                    t.StartsWith("macro ", StringComparison.Ordinal)) { depth++; continue; }
                if ((t == "endfor" || t == "endif" || t == "endmacro" || t == "endset") && depth > 0) { depth--; continue; }
                if (depth > 0) continue;
                if (t == "endset")
                {
                    // Execute body into a temp StringBuilder with same env
                    var msb = new System.Text.StringBuilder();
                    Execute(tokens, bodyStart, i, env, msb, ctx);
                    env.Set(name, msb.ToString());
                    return i + 1;
                }
            }
            // No endset
            var msb2 = new System.Text.StringBuilder();
            Execute(tokens, bodyStart, end, env, msb2, ctx);
            env.Set(name, msb2.ToString());
            return end;
        }

        ctx.Errors.Add($"JINJA ERROR: unrecognized 'set' tag syntax: '{tag}'");
        return tagIdx + 1;
    }

    // Expression evaluator

    private static readonly Regex IsTestRe = new(
        @"^(.+?)\s+is\s+(not\s+)?(defined|none|string|boolean|mapping|sequence|iterable|true|false)$",
        RegexOptions.Singleline);

    /// <summary>
    /// Public entry point retained for tests and simple use. Has no template
    /// token context, so macros/capture are unreachable through it; ordinary
    /// expressions evaluate identically to the in-template path.
    /// </summary>
    public static object? Eval(string expr, JinjaEnv env, List<string>? errors = null)
        => Eval(expr, env, new Ctx([], errors ?? []));

    private static object? Eval(string expr, JinjaEnv env, Ctx ctx)
    {
        expr = expr.Trim();
        if (expr.Length == 0) return null;

        // strip outer parens — only when the outermost ( and ) actually match
        if (expr.StartsWith('(') && expr.EndsWith(')') && MatchingOuterParens(expr))
            return Eval(expr[1..^1], env, ctx);

        // Inline conditional: X if C else Y  (lowest precedence)
        int ifPos = FindInlineIf(expr, out int elsePos);
        if (ifPos >= 0)
        {
            var trueVal = Eval(expr[..ifPos], env, ctx);
            if (elsePos >= 0)
            {
                var cond = Eval(expr[(ifPos + 3)..elsePos], env, ctx);
                var falseVal = Eval(expr[(elsePos + 4)..], env, ctx);
                return IsTruthy(cond) ? trueVal : falseVal;
            }
            var condOnly = Eval(expr[(ifPos + 3)..], env, ctx);
            return IsTruthy(condOnly) ? trueVal : (object)"";
        }

        // or / and  (lower precedence than not, so evaluated first)
        int orPos = FindKeywordOutside(expr, "or");
        if (orPos >= 0)
            return (object)(IsTruthy(Eval(expr[..orPos], env, ctx)) ||
                            IsTruthy(Eval(expr[(orPos + 2)..], env, ctx)));

        int andPos = FindKeywordOutside(expr, "and");
        if (andPos >= 0)
            return (object)(IsTruthy(Eval(expr[..andPos], env, ctx)) &&
                            IsTruthy(Eval(expr[(andPos + 3)..], env, ctx)));

        // leading not (binds tighter than and/or, looser than is-tests)
        if (expr.StartsWith("not ", StringComparison.Ordinal))
            return (object)!IsTruthy(Eval(expr[4..], env, ctx));

        // is-tests (full expression ending in a test keyword)
        var isM = IsTestRe.Match(expr);
        if (isM.Success)
        {
            string subject = isM.Groups[1].Value.Trim();
            bool neg = isM.Groups[2].Success;
            string test = isM.Groups[3].Value;
            object? v;
            bool r;
            if (test == "defined")
            {
                r = IsPlainWord(subject) ? env.ContainsKey(subject) : Eval(subject, env, ctx) != null;
            }
            else
            {
                v = Eval(subject, env, ctx);
                r = test switch
                {
                    "none" => v == null || (v is string sn && sn.Length == 0),
                    "string" => v is string,
                    "boolean" => v is bool,
                    "mapping" => v is JinjaDict,
                    "sequence" => v is string || v is List<object?> || v is List<JinjaDict>,
                    "iterable" => v is string || v is List<object?> || v is List<JinjaDict> || v is JinjaDict,
                    "true" => IsTruthy(v),
                    "false" => !IsTruthy(v),
                    _ => false
                };
            }
            return (object)(neg ? !r : r);
        }

        // == / !=
        int eqPos = FindSeqOutside(expr, "==");
        if (eqPos >= 0)
            return (object)(Stringify(Eval(expr[..eqPos], env, ctx)) == Stringify(Eval(expr[(eqPos + 2)..], env, ctx)));

        int neqPos = FindSeqOutside(expr, "!=");
        if (neqPos >= 0)
            return (object)(Stringify(Eval(expr[..neqPos], env, ctx)) != Stringify(Eval(expr[(neqPos + 2)..], env, ctx)));

        // in / not in
        int inPos = FindKeywordOutside(expr, "in");
        if (inPos >= 0)
        {
            string leftRaw = expr[..inPos];
            string leftTrim = leftRaw.TrimEnd();
            bool negIn = leftTrim.EndsWith("not", StringComparison.Ordinal);
            if (negIn) leftRaw = leftTrim[..^3];
            object? lhs = Eval(leftRaw, env, ctx);
            object? rhs = Eval(expr[(inPos + 2)..], env, ctx);
            bool contains = ContainsValue(rhs, lhs);
            return (object)(negIn ? !contains : contains);
        }

        // %  >  <  >=  <=
        int modPos = FindSeqOutside(expr, "%");
        if (modPos >= 0)
        {
            object? a = Eval(expr[..modPos], env, ctx);
            object? b = Eval(expr[(modPos + 1)..], env, ctx);
            if (TryLong(a, out long la) && TryLong(b, out long lb) && lb != 0)
                return (object)(la % lb);
            return (object)Stringify(a);
        }

        int gePos = FindSeqOutside(expr, ">=");
        if (gePos >= 0)
            return (object)(ToNumber(Eval(expr[..gePos], env, ctx)) >= ToNumber(Eval(expr[(gePos + 2)..], env, ctx)));
        int lePos = FindSeqOutside(expr, "<=");
        if (lePos >= 0)
            return (object)(ToNumber(Eval(expr[..lePos], env, ctx)) <= ToNumber(Eval(expr[(lePos + 2)..], env, ctx)));
        int gtPos = FindSeqOutside(expr, ">");
        if (gtPos >= 0)
            return (object)(ToNumber(Eval(expr[..gtPos], env, ctx)) > ToNumber(Eval(expr[(gtPos + 1)..], env, ctx)));
        int ltPos = FindSeqOutside(expr, "<");
        if (ltPos >= 0)
            return (object)(ToNumber(Eval(expr[..ltPos], env, ctx)) < ToNumber(Eval(expr[(ltPos + 1)..], env, ctx)));

        // +
        int plusPos = FindSeqOutside(expr, "+");
        if (plusPos >= 0)
        {
            object? a = Eval(expr[..plusPos], env, ctx);
            object? b = Eval(expr[(plusPos + 1)..], env, ctx);
            if (TryLong(a, out long la2) && TryLong(b, out long lb2))
                return (object)(la2 + lb2);
            return (object)(Stringify(a) + Stringify(b));
        }

        // - (binary; ignore leading minus)
        int minusPos = FindSeqOutside(expr, "-");
        if (minusPos > 0)
        {
            object? a = Eval(expr[..minusPos], env, ctx);
            object? b = Eval(expr[(minusPos + 1)..], env, ctx);
            if (TryLong(a, out long la3) && TryLong(b, out long lb3))
                return (object)(la3 - lb3);
        }

        // filters (a | f | g(args) ...)
        int pipePos = FindTopLevelPipe(expr);
        if (pipePos >= 0)
        {
            object? val = Eval(expr[..pipePos], env, ctx);
            string rest = expr[(pipePos + 1)..];
            foreach (string seg in SplitTopLevel(rest, '|'))
            {
                string spec = seg.Trim();
                if (spec.Length == 0) continue;
                val = ApplyFilter(spec, val, env, ctx);
            }
            return val;
        }

        // Literals
        if (expr is "none" or "None") return null;
        if (expr is "true" or "True") return (object)true;
        if (expr is "false" or "False") return (object)false;

        // String literal (possibly concatenated already handled by +)
        if ((expr.StartsWith('\'') && expr.EndsWith('\'')) ||
            (expr.StartsWith('"') && expr.EndsWith('"')))
            return UnescapeString(expr[1..^1]);

        if (TryLong(expr, out long intVal)) return (object)intVal;

        // List literal
        if (expr.StartsWith('[') && expr.EndsWith(']') && MatchingBrackets(expr))
        {
            var list = new List<object?>();
            string inner = expr[1..^1].Trim();
            if (inner.Length > 0)
                foreach (string el in SplitTopLevel(inner, ','))
                    list.Add(Eval(el, env, ctx));
            return list;
        }

        // Call: name(args) or obj.method(args)
        if (expr.EndsWith(')'))
        {
            int open = FindMatchingOpenParen(expr);
            if (open > 0)
            {
                string callee = expr[..open].Trim();
                string argStr = expr[(open + 1)..^1];
                var rawArgs = SplitTopLevel(argStr, ',')
                    .Select(a => a.Trim()).Where(a => a.Length > 0).ToList();

                int dot = LastTopLevelDot(callee);
                if (dot > 0)
                {
                    string objExpr = callee[..dot].Trim();
                    string method = callee[(dot + 1)..].Trim();
                    object? obj = Eval(objExpr, env, ctx);
                    var args = rawArgs.Select(a => Eval(a, env, ctx)).ToList();
                    return InvokeMethod(method, obj, args, ctx);
                }

                // macro?
                if (ctx.Macros.TryGetValue(callee, out var def))
                    return InvokeMacro(def, rawArgs, env, ctx);

                // builtin function
                return InvokeBuiltin(callee, rawArgs, env, ctx);
            }
        }

        // Bracket indexing / slicing
        if (expr.EndsWith(']'))
        {
            int openB = FindMatchingOpenBracket(expr);
            if (openB > 0)
            {
                object? obj = Eval(expr[..openB], env, ctx);
                string idx = expr[(openB + 1)..^1].Trim();
                return AccessBracket(obj, idx, env, ctx);
            }
        }

        // dot field access
        int d = FindTopLevelDot(expr);
        if (d > 0)
        {
            object? obj = Eval(expr[..d], env, ctx);
            return AccessField(obj, expr[(d + 1)..].Trim(), ctx);
        }

        // Plain identifier
        var result = env.Get(expr);
        if (result is null && !env.ContainsKey(expr))
        {
            string msg = $"JINJA ERROR: variable '{expr}' not found in template env";
            ctx.Errors.Add(msg);
            SanityChecks.WriteLine($"JinjaTemplateFormatter: variable '{expr}' not found in template env");
        }
        return result;
    }

    // Call helpers

    private static object? InvokeBuiltin(string name, List<string> rawArgs, JinjaEnv env, Ctx ctx)
    {
        switch (name)
        {
            case "range":
            {
                var vals = rawArgs.Select(a => (long)ToNumber(Eval(a, env, ctx))).ToList();
                var list = new List<object?>();
                if (vals.Count == 1)
                    for (long i = 0; i < vals[0]; i++) list.Add(i);
                else if (vals.Count >= 2)
                    for (long i = vals[0]; i < vals[1]; i++) list.Add(i);
                return list;
            }
            case "strftime_now":
            {
                string fmt = rawArgs.Count > 0 ? rawArgs[0].Trim('"', '\'', ' ') : "";
                try { return (object)(fmt.Length > 0 ? DateTime.Now.ToString(fmt) : DateTime.Now.ToString("dd MMM yyyy")); }
                catch { return (object)DateTime.Now.ToString("dd MMM yyyy"); }
            }
            case "raise_exception":
                return (object)"";
            default:
                ctx.Errors.Add($"JINJA ERROR: unknown function '{name}'");
                return null;
        }
    }

    private static object? InvokeMacro(MacroDef def, List<string> rawArgs, JinjaEnv env, Ctx ctx)
    {
        var macEnv = env.Push();
        var positional = new List<object?>();
        var named = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (string a in rawArgs)
        {
            int eq = FindTopLevelEq(a);
            if (eq > 0 && IsPlainWord(a[..eq].Trim()))
                named[a[..eq].Trim()] = Eval(a[(eq + 1)..], env, ctx);
            else
                positional.Add(Eval(a, env, ctx));
        }

        int pi = 0;
        foreach (var p in def.Params)
        {
            object? val;
            if (named.TryGetValue(p.Name, out var nv)) val = nv;
            else if (pi < positional.Count) val = positional[pi++];
            else if (p.DefaultExpr is not null) val = Eval(p.DefaultExpr, macEnv, ctx);
            else val = null;
            macEnv.Set(p.Name, val ?? (object)"");
        }

        var sb = new StringBuilder();
        Execute(ctx.Tokens, def.BodyStart, def.BodyEnd, macEnv, sb, ctx);
        return sb.ToString();
    }

    private static object? InvokeMethod(string method, object? obj, List<object?> args, Ctx ctx)
    {
        switch (method)
        {
            case "get":
            {
                string key = args.Count > 0 ? Stringify(args[0]) : "";
                object? def = args.Count > 1 ? args[1] : null;
                if (obj is JinjaDict d && d.TryGetValue(key, out var v)) return v;
                return def;
            }
            case "split":
            {
                string s = obj as string ?? Stringify(obj);
                string sep = args.Count > 0 ? Stringify(args[0]) : " ";
                return s.Split(sep).Select(x => (object?)x).ToList();
            }
            case "strip":
                return (obj as string ?? Stringify(obj)).Trim();
            case "startswith":
                return (object)(obj as string ?? Stringify(obj)).StartsWith(args.Count > 0 ? Stringify(args[0]) : "", StringComparison.Ordinal);
            case "endswith":
                return (object)(obj as string ?? Stringify(obj)).EndsWith(args.Count > 0 ? Stringify(args[0]) : "", StringComparison.Ordinal);
            case "keys":
                return obj is JinjaDict kd ? kd.Select(kv => (object?)kv.Key).ToList() : new List<object?>();
            case "values":
                return obj is JinjaDict vd ? vd.Select(kv => kv.Value).ToList() : new List<object?>();
            case "items":
                return obj is JinjaDict id ? id.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList() : new List<object?>();
            default:
                ctx.Errors.Add($"JINJA ERROR: unknown method '{method}'");
                return null;
        }
    }

    private static object? ApplyFilter(string spec, object? val, JinjaEnv env, Ctx ctx)
    {
        string name = spec;
        var rawArgs = new List<string>();
        int paren = spec.IndexOf('(');
        if (paren >= 0 && spec.EndsWith(')'))
        {
            name = spec[..paren].Trim();
            string argStr = spec[(paren + 1)..^1];
            if (argStr.Trim().Length > 0)
                rawArgs = SplitTopLevel(argStr, ',').Select(a => a.Trim()).Where(a => a.Length > 0).ToList();
        }
        name = name.Trim();

        switch (name)
        {
            case "trim":
                return (val as string ?? Stringify(val)).Trim();
            case "lower":
                return (val as string ?? Stringify(val)).ToLowerInvariant();
            case "upper":
                return (val as string ?? Stringify(val)).ToUpperInvariant();
            case "string":
                return Stringify(val);
            case "length":
                return val switch
                {
                    string s => (object)(long)s.Length,
                    List<object?> l => (object)(long)l.Count,
                    List<JinjaDict> l => (object)(long)l.Count,
                    JinjaDict d => (object)(long)d.Count,
                    _ => (object)0L
                };
            case "list":
                return val switch
                {
                    List<object?> l => l,
                    List<JinjaDict> l => l.Cast<object?>().ToList(),
                    JinjaDict d => d.Select(kv => (object?)kv.Key).ToList(),
                    null => new List<object?>(),
                    _ => new List<object?> { val }
                };
            case "first":
                var fl = AsIterableList(val);
                return fl is { Count: > 0 } ? fl[0] : null;
            case "last":
                var ll = AsIterableList(val);
                return ll is { Count: > 0 } ? ll[^1] : null;
            case "join":
            {
                string sep = rawArgs.Count > 0 ? Stringify(Eval(rawArgs[0], env, ctx)) : "";
                var jl = AsIterableList(val);
                return jl is null ? Stringify(val) : string.Join(sep, jl.Select(Stringify));
            }
            case "default":
            {
                bool useDefault = val is null || (val is string ds && ds.Length == 0);
                if (rawArgs.Count >= 2)
                {
                    object? second = Eval(rawArgs[1], env, ctx);
                    if (IsTruthy(second)) useDefault = val is null;
                }
                return useDefault && rawArgs.Count > 0 ? (Eval(rawArgs[0], env, ctx) ?? (object)"") : val;
            }
            case "dictsort":
            {
                if (val is not JinjaDict dd) return val;
                return dd.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                         .Select(kv => (object?)new List<object?> { kv.Key, kv.Value })
                         .ToList();
            }
            case "map":
            {
                var ml = AsIterableList(val);
                if (ml is null) return val;
                string attr = rawArgs.Count > 0 ? rawArgs[0].Trim().Trim('"', '\'') : "";
                var outList = new List<object?>();
                foreach (var el in ml)
                    outList.Add(attr switch
                    {
                        "upper" => (el as string ?? Stringify(el)).ToUpperInvariant(),
                        "lower" => (el as string ?? Stringify(el)).ToLowerInvariant(),
                        "trim" => (el as string ?? Stringify(el)).Trim(),
                        "string" => Stringify(el),
                        _ => el
                    });
                return outList;
            }
            case "tojson":
                return ToJson(val);
            case "escape" or "e":
                return Stringify(val);
            default:
                ctx.Errors.Add($"JINJA ERROR: unknown filter '{name}'");
                return val;
        }
    }

    // Index / field access

    private static object? AccessBracket(object? obj, string idx, JinjaEnv env, Ctx ctx)
    {
        if (idx.Contains(':'))
        {
            var parts = idx.Split(':');
            int? from = parts[0].Trim().Length > 0 ? (int)ToNumber(Eval(parts[0], env, ctx)) : null;
            int? to = parts.Length > 1 && parts[1].Trim().Length > 0 ? (int)ToNumber(Eval(parts[1], env, ctx)) : null;
            if (obj is string sSlice)
            {
                int start = from ?? 0;
                int finish = to ?? sSlice.Length;
                if (start < 0) start += sSlice.Length;
                if (finish < 0) finish += sSlice.Length;
                start = Math.Clamp(start, 0, sSlice.Length);
                finish = Math.Clamp(finish, 0, sSlice.Length);
                return finish >= start ? sSlice[start..finish] : "";
            }
            var sliceList = AsIterableList(obj);
            if (sliceList is not null)
            {
                int start = from ?? 0;
                int finish = to ?? sliceList.Count;
                if (start < 0) start += sliceList.Count;
                if (finish < 0) finish += sliceList.Count;
                start = Math.Clamp(start, 0, sliceList.Count);
                finish = Math.Clamp(finish, 0, sliceList.Count);
                var outl = new List<object?>();
                for (int i = start; i < finish; i++) outl.Add(sliceList[i]);
                return outl;
            }
            return null;
        }

        object? key = Eval(idx, env, ctx);
        if (obj is JinjaDict dict)
            return dict.TryGetValue(Stringify(key), out var dv) ? dv : null;
        if (obj is string str)
        {
            long i = TryLong(key, out long ki) ? ki : 0;
            if (i < 0) i += str.Length;
            return i >= 0 && i < str.Length ? (object)str[(int)i].ToString() : "";
        }
        var list = AsIterableList(obj);
        if (list is not null)
        {
            long i = TryLong(key, out long ki2) ? ki2 : 0;
            if (i < 0) i += list.Count;
            return i >= 0 && i < list.Count ? list[(int)i] : null;
        }
        return null;
    }

    private static object? AccessField(object? obj, string field, Ctx ctx)
    {
        return obj switch
        {
            JinjaNamespace ns => ns.Get(field),
            JinjaDict d when d.TryGetValue(field, out var v) => v,
            _ => null
        };
    }

    private static int FindTopLevelPipe(string s)
    {
        bool inS = false, inD = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') depth--;
                else if (c == '|' && depth == 0 && i + 1 < s.Length && s[i + 1] != '|' && (i == 0 || s[i - 1] != '|'))
                    return i;
            }
        }
        return -1;
    }

    private static bool ContainsValue(object? container, object? item)
    {
        switch (container)
        {
            case string s:
                return s.Contains(Stringify(item), StringComparison.Ordinal);
            case List<object?> l:
                return l.Any(x => Stringify(x) == Stringify(item));
            case List<JinjaDict> l:
                return l.Any(x => Stringify(x) == Stringify(item));
            case JinjaDict d:
                return d.ContainsKey(Stringify(item));
            default:
                return false;
        }
    }

    private static List<string> SplitTopLevel(string s, char sep)
    {
        var parts = new List<string>();
        bool inS = false, inD = false;
        int depth = 0, last = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') depth--;
                else if (c == sep && depth == 0) { parts.Add(s[last..i]); last = i + 1; }
            }
        }
        parts.Add(s[last..]);
        return parts;
    }

    private static int FindMatchingOpenParen(string s)
    {
        int depth = 0;
        bool inS = false, inD = false;
        for (int i = s.Length - 1; i >= 0; i--)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c == ')') depth++;
                else if (c == '(') { depth--; if (depth == 0) return i; }
            }
        }
        return -1;
    }

    private static int FindMatchingOpenBracket(string s)
    {
        int depth = 0;
        bool inS = false, inD = false;
        for (int i = s.Length - 1; i >= 0; i--)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c == ']') depth++;
                else if (c == '[') { depth--; if (depth == 0) return i; }
            }
        }
        return -1;
    }

    private static int LastTopLevelDot(string s)
    {
        int found = -1, depth = 0;
        bool inS = false, inD = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') depth--;
                else if (c == '.' && depth == 0) found = i;
            }
        }
        return found;
    }

    private static bool MatchingBrackets(string s)
    {
        int depth = 0;
        bool inS = false, inD = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) inS = !inS;
            else if (c == '"' && !inS) inD = !inD;
            else if (!inS && !inD)
            {
                if (c == '[') depth++;
                else if (c == ']') { depth--; if (depth == 0 && i != s.Length - 1) return false; }
            }
        }
        return depth == 0;
    }

    private static bool IsPlainWord(string s) => s.All(ch => char.IsLetterOrDigit(ch) || ch == '_');


    // Additional scan/value helpers used by the reworked evaluator

    private static int FindSeqOutside(string s, string op)
    {
        bool inS = false, inD = false;
        int depth = 0;
        for (int i = 0; i + op.Length <= s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) { inS = !inS; continue; }
            if (c == '"' && !inS) { inD = !inD; continue; }
            if (inS || inD) continue;
            if (c is '(' or '[' or '{') { depth++; continue; }
            if (c is ')' or ']' or '}') { depth--; continue; }
            if (depth != 0) continue;
            if (string.CompareOrdinal(s, i, op, 0, op.Length) == 0)
            {
                if (op == ">" && i + 1 < s.Length && s[i + 1] == '=') continue;
                if (op == "<" && i + 1 < s.Length && s[i + 1] == '=') continue;
                return i;
            }
        }
        return -1;
    }

    private static int FindTopLevelDot(string s)
    {
        bool inS = false, inD = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) { inS = !inS; continue; }
            if (c == '"' && !inS) { inD = !inD; continue; }
            if (inS || inD) continue;
            if (c is '(' or '[' or '{') { depth++; continue; }
            if (c is ')' or ']' or '}') { depth--; continue; }
            if (depth == 0 && c == '.') return i;
        }
        return -1;
    }

    private static int FindTopLevelEq(string s)
    {
        bool inS = false, inD = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) { inS = !inS; continue; }
            if (c == '"' && !inS) { inD = !inD; continue; }
            if (inS || inD) continue;
            if (c is '(' or '[' or '{') { depth++; continue; }
            if (c is ')' or ']' or '}') { depth--; continue; }
            if (depth != 0 || c != '=') continue;
            char prev = i > 0 ? s[i - 1] : ' ';
            char next = i + 1 < s.Length ? s[i + 1] : ' ';
            if (prev is '=' or '!' or '<' or '>' || next is '=') continue;
            return i;
        }
        return -1;
    }

    /// <summary>
    /// Finds the top-level <c>if</c> of an inline conditional <c>X if C else Y</c>,
    /// returning its index and (via <paramref name="elsePos"/>) the index of the
    /// matching <c>else</c> (or -1). Scans from index 0 so a leading string
    /// literal cannot desynchronise quote tracking.
    /// </summary>
    private static int FindInlineIf(string s, out int elsePos)
    {
        elsePos = -1;
        int ifPos = -1;
        bool inS = false, inD = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inD) { inS = !inS; continue; }
            if (c == '"' && !inS) { inD = !inD; continue; }
            if (inS || inD) continue;
            if (c is '(' or '[' or '{') { depth++; continue; }
            if (c is ')' or ']' or '}') { depth--; continue; }
            if (depth != 0) continue;

            bool atWord = c == ' ' && i + 3 < s.Length && s[i + 3] == ' ';
            if (ifPos < 0 && atWord && s[i + 1] == 'i' && s[i + 2] == 'f')
            {
                ifPos = i + 1;
                i += 2;
                continue;
            }
            if (ifPos >= 0 && c == ' ' && i + 4 < s.Length &&
                s[i + 1] == 'e' && s[i + 2] == 'l' && s[i + 3] == 's' && s[i + 4] == 'e' &&
                (i + 5 >= s.Length || s[i + 5] == ' '))
            {
                elsePos = i + 1;
                break;
            }
        }
        return ifPos;
    }

    private static bool TryLong(object? v, out long r)
    {
        switch (v)
        {
            case long l: r = l; return true;
            case int i: r = i; return true;
            case double d: r = (long)d; return true;
            case string s when long.TryParse(s.Trim(), out long p): r = p; return true;
            default: r = 0; return false;
        }
    }

    private static double ToNumber(object? v) => v switch
    {
        long l => l,
        int i => i,
        double d => d,
        bool b => b ? 1 : 0,
        string s when double.TryParse(s.Trim(), out double p) => p,
        _ => 0
    };

    private static string ToJson(object? v) => ObjectToJsonValue(v);

    private static string UnescapeString(string s)
    {
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '\\' && i + 1 < s.Length)
            {
                char n = s[i + 1];
                switch (n)
                {
                    case 'n': sb.Append('\n'); i++; break;
                    case 't': sb.Append('\t'); i++; break;
                    case 'r': sb.Append('\r'); i++; break;
                    case '\\': sb.Append('\\'); i++; break;
                    case '\'': sb.Append('\''); i++; break;
                    case '"': sb.Append('"'); i++; break;
                    default: sb.Append(n); i++; break;
                }
            }
            else sb.Append(s[i]);
        }
        return sb.ToString();
    }

    // Helpers

    private static int FindMatchingEnd(
        List<Token> tokens, int start, int end,
        string openKeyword, string closeKeyword)
    {
        int depth = 0;
        for (int i = start + 1; i < end; i++)
        {
            if (tokens[i].Kind != TKind.Tag) continue;
            string t = tokens[i].Body;
            if (t.StartsWith(openKeyword + " ", StringComparison.Ordinal) ||
                t == openKeyword) depth++;
            else if (t == closeKeyword)
            {
                if (depth == 0) return i;
                depth--;
            }
        }
        return end;
    }

    /// <summary>
    /// Finds a whole-word keyword (<c>and</c>, <c>or</c>) outside string quotes
    /// and brackets.  Returns the index of the space before the keyword, or -1.
    /// The keyword must be surrounded by whitespace and at depth 0.
    /// </summary>
    private static int FindKeywordOutside(string s, string kw)
    {
        bool inSingle = false, inDouble = false;
        int depth = 0;
        int kwLen = kw.Length;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == ' ' && i + kwLen + 1 < s.Length
                         && s[(i + 1)..(i + 1 + kwLen)] == kw
                         && s[i + 1 + kwLen] == ' ')
                    return i + 1; // points at first char of keyword
            }
        }
        return -1;
    }

    /// <summary>
    /// Returns true when the outermost <c>(</c> (position 0) is matched by the
    /// innermost <c>)</c> (position <c>s.Length-1</c>), i.e. the parens around
    /// the whole expression truly belong together as a single grouping pair.
    /// </summary>
    private static bool MatchingOuterParens(string s)
    {
        int depth = 0;
        bool inSingle = false, inDouble = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) { inSingle = !inSingle; continue; }
            if (c == '"' && !inSingle) { inDouble = !inDouble; continue; }
            if (inSingle || inDouble) continue;
            if (c == '(') depth++;
            else if (c == ')')
            {
                depth--;
                // Closing paren at depth 0 before the end means the outermost
                // opening ( was NOT matched by the final character.
                if (depth == 0 && i < s.Length - 1) return false;
            }
        }
        return depth == 0;
    }

    public static bool IsTruthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        string s => s.Length > 0,
        List<JinjaDict> l => l.Count > 0,
        _ => true
    };

    private static string Stringify(object? v) => v switch
    {
        null => "",
        bool b => b ? "True" : "False",
        _ => v.ToString() ?? ""
    };

    private static string RoleName(ChatRole role) => role switch
    {
        ChatRole.System => "system",
        ChatRole.Agent => "assistant",
        ChatRole.User => "user",
        ChatRole.Tool => "tool",
        _ => "user"
    };

    private static List<JinjaDict> JsonToJinjaDictList(string? json)
    {
        var result = new List<JinjaDict>();
        if (string.IsNullOrWhiteSpace(json)) return result;
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != System.Text.Json.JsonValueKind.Array) return result;
            foreach (var el in doc.RootElement.EnumerateArray())
            {
                var dict = JsonElementToDict(el);
                if (dict is not null) result.Add(dict);
            }
        }
        catch { /* malformed tool JSON */ }
        return result;
    }

    private static JinjaDict? JsonElementToDict(System.Text.Json.JsonElement el)
    {
        if (el.ValueKind != System.Text.Json.JsonValueKind.Object) return null;
        var dict = new JinjaDict();
        foreach (var prop in el.EnumerateObject())
        {
            dict[prop.Name] = JsonElementToObject(prop.Value);
        }
        return dict;
    }

    private static object? JsonElementToObject(System.Text.Json.JsonElement el) => el.ValueKind switch
    {
        System.Text.Json.JsonValueKind.String => el.GetString(),
        System.Text.Json.JsonValueKind.Number => el.TryGetInt64(out var l) ? (object)l : (object)el.GetDouble(),
        System.Text.Json.JsonValueKind.True => (object)true,
        System.Text.Json.JsonValueKind.False => (object)false,
        System.Text.Json.JsonValueKind.Null => null,
        System.Text.Json.JsonValueKind.Object => JsonElementToDict(el),
        System.Text.Json.JsonValueKind.Array => JsonArrayToList(el),
        _ => el.GetRawText()
    };

    private static List<object?> JsonArrayToList(System.Text.Json.JsonElement el)
    {
        var list = new List<object?>();
        foreach (var item in el.EnumerateArray()) list.Add(JsonElementToObject(item));
        return list;
    }

    private static string DictToJson(JinjaDict dict)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var kv in dict)
        {
            if (!first) sb.Append(", ");
            first = false;
            sb.Append('"').Append(kv.Key).Append("\": ");
            sb.Append(ObjectToJsonValue(kv.Value));
        }
        sb.Append('}');
        return sb.ToString();
    }

    private static string ListToJson(List<JinjaDict> list)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < list.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(DictToJson(list[i]));
        }
        sb.Append(']');
        return sb.ToString();
    }

    private static string ObjectToJsonValue(object? v) => v switch
    {
        null => "null",
        string s => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"",
        bool b => b ? "true" : "false",
        long l => l.ToString(),
        int i => i.ToString(),
        double d => d.ToString("G"),
        JinjaDict d => DictToJson(d),
        List<object?> l => ListToJsonValue(l),
        _ => "\"" + (v.ToString() ?? "")?.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""
    };

    private static string ListToJsonValue(List<object?> list)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < list.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(ObjectToJsonValue(list[i]));
        }
        sb.Append(']');
        return sb.ToString();
    }
}
