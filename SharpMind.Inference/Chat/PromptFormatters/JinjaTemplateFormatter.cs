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
        bool hasTools = toolsList.Count > 0;

        var env = new JinjaEnv();
        env.Set("messages", messages);
        env.Set("bos_token", bosToken);
        env.Set("eos_token", eosToken);
        env.Set("add_generation_prompt", (object)true);  // always true for inference
        env.Set("enable_thinking", (object)enableThinking);
        // Only set tools/custom_tools when there are actual tool definitions.
        // An empty list makes templates think tools ARE present (is not none → true),
        // which causes Llama-3 to render tool-related system message content
        // even when no tools were configured.
        if (hasTools)
        {
            env.Set("tools", (object)toolsList);
            env.Set("custom_tools", (object)toolsList);
        }
        env.Set("tools_in_user_message", (object)false); // Llama-3: tool calls not in user message
        env.Set("date_string", (object)DateTime.Now.ToString("dd MMM yyyy"));
        env.Set("strftime_now", (object)DateTime.Now.ToString("dd MMM yyyy"));

        var errors = new List<string>();
        var tokens = Tokenise(_template);
        var sb = new StringBuilder();
        Execute(tokens, 0, tokens.Count, env, sb, errors);
        LastErrors = errors.Count > 0 ? errors : null;
        return sb.ToString();
    }

    // Tokeniser
    // Splits the template into a flat list of Token objects.

    private enum TKind { Text, Output, Tag }

    private sealed record Token(TKind Kind, string Body, bool StripLeft, bool StripRight);

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
        JinjaEnv env, StringBuilder sb, List<string> errors)
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
                var val = Eval(tok.Body, env, errors);
                if (val != null) sb.Append(Stringify(val));
                i++;
                continue;
            }

            // Tag
            string tag = tok.Body;

            if (tag.StartsWith("for ", StringComparison.Ordinal))
            {
                i = ExecuteFor(tokens, i, end, env, sb, errors);
                continue;
            }

            if (tag.StartsWith("if ", StringComparison.Ordinal) || tag == "if")
            {
                i = ExecuteIf(tokens, i, end, env, sb, errors);
                continue;
            }

            if (tag.StartsWith("set ", StringComparison.Ordinal))
            {
                ExecuteSet(tag, env, errors);
                i++;
                continue;
            }

            // Unrecognized tag — log error and skip
            errors.Add($"Unrecognized Jinja tag at token {i}: '{tag}'");
            i++;
        }
    }

    // For loop

    private static int ExecuteFor(
        List<Token> tokens, int forIdx, int end,
        JinjaEnv env, StringBuilder sb, List<string> errors)
    {
        // Parse "for VAR in EXPR"
        var m = RegexGenerated.JinjaForVarInExpr.Match(tokens[forIdx].Body);// Regex.Match(tokens[forIdx].Body, @"^for\s+(\w+)\s+in\s+(.+)$");
        if (!m.Success)
        {
            errors.Add($"JINJA ERROR: failed to parse 'for' tag at token {forIdx}: '{tokens[forIdx].Body}'");
            return forIdx + 1;
        }

        string varName = m.Groups[1].Value;
        string iterExpr = m.Groups[2].Value.Trim();
        var iterable = Eval(iterExpr, env, errors);

        // Find matching endfor
        int bodyStart = forIdx + 1;
        int bodyEnd = FindMatchingEnd(tokens, forIdx, end, "for", "endfor");

        if (iterable is List<JinjaDict> list)
        {
            int count = list.Count;
            for (int idx = 0; idx < count; idx++)
            {
                var child = env.Push();
                var item = list[idx];
                child.Set(varName, item);

                var loop = new JinjaDict
                {
                    ["first"] = (object)(idx == 0),
                    ["last"] = (object)(idx == count - 1),
                    ["index0"] = (object)(long)idx,
                    ["index"] = (object)(long)(idx + 1)
                };
                child.Set("loop", loop);

                Execute(tokens, bodyStart, bodyEnd, child, sb, errors);
            }
        }
        else
        {
            errors.Add($"JINJA ERROR: 'for {varName} in {iterExpr}' — expression is not a list (got {iterable?.GetType().Name ?? "null"})");
        }

        return bodyEnd + 1; // +1 to skip the endfor token
    }

    // If / elif / else

    private static int ExecuteIf(
        List<Token> tokens, int ifIdx, int end,
        JinjaEnv env, StringBuilder sb, List<string> errors)
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
                    if (cond == null || IsTruthy(Eval(cond, env, errors)))
                    {
                        Execute(tokens, bs, be, env, sb, errors);
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
            if (cond == null || IsTruthy(Eval(cond, env, errors)))
            {
                Execute(tokens, bs, be, env, sb, errors);
                break;
            }
        }
        return end;
    }

    // Set

    private static void ExecuteSet(string tag, JinjaEnv env, List<string> errors)
    {
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
                ns.Set(k, Eval(v, env, errors) ?? (object)"");
            }
            env.Set(name, ns);
            return;
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
                jns.Set(field, Eval(expr, env, errors) ?? (object)"");
            else if (target is JinjaDict jd)
                jd[field] = Eval(expr, env, errors) ?? (object)"";
            else
                errors.Add($"JINJA ERROR: 'set {obj}.{field} = {expr}' — '{obj}' is not a namespace or dict (got {target?.GetType().Name ?? "null"})");
            return;
        }

        // set var = expr
        var simple = RegexGenerated.JinjaSetVarEqExpr.Match(tag);// Regex.Match(tag, @"^set\s+(\w+)\s*=\s*(.+)$");
        if (simple.Success)
        {
            string name = simple.Groups[1].Value;
            string expr = simple.Groups[2].Value.Trim();
            env.Set(name, Eval(expr, env, errors) ?? (object)"");
            return;
        }

        errors.Add($"JINJA ERROR: unrecognized 'set' tag syntax: '{tag}'");
    }

    // Expression evaluator

    public static object? Eval(string expr, JinjaEnv env, List<string>? errors = null)
    {
        expr = expr.Trim();

        // strip outer parens — only when the outermost ( and ) actually match
        // (avoid falsely stripping when the top-level expression uses parens
        //  for grouping sub-expressions, e.g. (A or B) and (C or D))
        if (expr.StartsWith('(') && expr.EndsWith(')') && MatchingOuterParens(expr))
            return Eval(expr[1..^1], env, errors);

        // not X is defined
        if (expr.StartsWith("not ", StringComparison.Ordinal))
        {
            var rest = expr[4..].TrimStart();
            var restM = RegexGenerated.JinjaIsDefined.Match(rest);
            if (restM.Success)
            {
                bool isDef = env.ContainsKey(restM.Groups[1].Value);
                return (object)!isDef;
            }
        }

        // X is defined / X is not defined  (general expression, not just \w+)
        var isDefinedM = RegexGenerated.JinjaIsDefined.Match(expr);
        if (isDefinedM.Success)
        {
            string target = isDefinedM.Groups[1].Value;
            // Use env.ContainsKey directly to avoid logging errors for undefined variables.
            bool isDef = env.ContainsKey(target);
            return (object)(isDefinedM.Groups[2].Success ? !isDef : isDef);
        }

        // X is none / X is not none
        var noneM = RegexGenerated.JinjaXIsNoneNotNone.Match(expr);// Regex.Match(expr, @"^(.+?)\s+is\s+(not\s+)?none$");
        if (noneM.Success)
        {
            var v = Eval(noneM.Groups[1].Value, env, errors);
            bool isNone = v == null || v is string s && s == "";
            return (object)(noneM.Groups[2].Success ? !isNone : isNone);
        }

        // X is true / X is false / X is not true / X is not false
        var boolM = RegexGenerated.JinjaIsBool.Match(expr);
        if (boolM.Success)
        {
            var v = Eval(boolM.Groups[1].Value, env, errors);
            bool target = boolM.Groups[3].Value == "true";
            bool actual = v is bool vb ? vb : IsTruthy(v);
            bool isMatch = actual == target;
            return (object)(boolM.Groups[2].Success ? !isMatch : isMatch);
        }

        // X is string / X is not string
        var isStringM = RegexGenerated.JinjaIsString.Match(expr);
        if (isStringM.Success)
        {
            var v = Eval(isStringM.Groups[1].Value, env, errors);
            bool isStr = v is string;
            return (object)(isStringM.Groups[2].Success ? !isStr : isStr);
        }

        // X is iterable / X is not iterable
        var isIterableM = RegexGenerated.JinjaIsIterable.Match(expr);
        if (isIterableM.Success)
        {
            var v = Eval(isIterableM.Groups[1].Value, env, errors);
            bool isIter = v is string || v is System.Collections.IList || v is System.Collections.ICollection;
            return (object)(isIterableM.Groups[2].Success ? !isIter : isIter);
        }

        // not X  /  not(X) (function-call style)
        if (expr.StartsWith("not ", StringComparison.Ordinal))
            return (object)!IsTruthy(Eval(expr[4..], env, errors));
        if (expr.StartsWith("not(", StringComparison.Ordinal))
        {
            int depth = 1;
            for (int i = 4; i < expr.Length; i++)
            {
                if (expr[i] == '(') depth++;
                else if (expr[i] == ')') { depth--; if (depth == 0) return (object)!IsTruthy(Eval(expr[4..i], env, errors)); }
            }
        }

        // 'literal' in X  (substring test)
        var inMatch = RegexGenerated.JinjaLiteralInXSubstr.Match(expr); // Regex.Match(expr, @"^'([^']*)'\s+in\s+(.+)$");
        if (inMatch.Success)
        {
            string needle = inMatch.Groups[1].Value;
            var hay = Eval(inMatch.Groups[2].Value, env, errors);
            string hayStr = hay is string hs ? hs : Stringify(hay);
            return (object)hayStr.Contains(needle, StringComparison.Ordinal);
        }

        // "literal" in X
        var inMatch2 = RegexGenerated.JinjaLiteralInX.Match(expr); //Regex.Match(expr, "^\"([^\"]*)\"\\s+in\\s+(.+)$");
        if (inMatch2.Success)
        {
            string needle = inMatch2.Groups[1].Value;
            var hay = Eval(inMatch2.Groups[2].Value, env, errors);
            string hayStr = hay is string hs ? hs : Stringify(hay);
            return (object)hayStr.Contains(needle, StringComparison.Ordinal);
        }

        // Operator precedence (lowest first, matching Jinja2 / Python):
        //   or  →  and  →  ==/!=  →  is/in/not  →  +  →  %  →  atom
        // Each level uses a bracket-aware search so that expressions like
        // messages[0]['role'] != 'system'  are not split at the wrong point.

        // X or Y
        int orPos = FindKeywordOutside(expr, "or");
        if (orPos >= 0)
        {
            var lv = Eval(expr[..orPos].TrimEnd(), env, errors);
            return IsTruthy(lv) ? lv : Eval(expr[(orPos + 2)..].TrimStart(), env, errors);
        }

        // X and Y
        int andPos = FindKeywordOutside(expr, "and");
        if (andPos >= 0)
            return (object)(
                IsTruthy(Eval(expr[..andPos].TrimEnd(), env, errors)) &&
                IsTruthy(Eval(expr[(andPos + 3)..].TrimStart(), env, errors)));

        // X == Y
        int eqPos = FindOperatorOutsideQuotes(expr, '=', requireDouble: true);
        if (eqPos >= 0)
        {
            var left = Eval(expr[..eqPos].TrimEnd(), env, errors);
            var right = Eval(expr[(eqPos + 2)..].TrimStart(), env, errors);
            return (object)(Stringify(left) == Stringify(right));
        }

        // X != Y
        int nePos = FindNeqOutside(expr);
        if (nePos >= 0)
        {
            var left = Eval(expr[..nePos].TrimEnd(), env, errors);
            var right = Eval(expr[(nePos + 2)..].TrimStart(), env, errors);
            return (object)(Stringify(left) != Stringify(right));
        }

        // X < Y, X > Y, X <= Y, X >= Y (comparison operators)
        int cmpPos = FindCmpOutside(expr, out string cmpOp);
        if (cmpPos >= 0)
        {
            var left = Eval(expr[..cmpPos].TrimEnd(), env, errors);
            var right = Eval(expr[(cmpPos + cmpOp.Length)..].TrimStart(), env, errors);
            long ln = left is long ll ? ll : left is int ii ? ii : 0;
            long rn = right is long rl ? rl : right is int ri ? ri : 0;
            return (object)(cmpOp switch
            {
                "<" => ln < rn,
                ">" => ln > rn,
                "<=" => ln <= rn,
                ">=" => ln >= rn,
                _ => false
            });
        }

        // Modulo: expr % expr (outside quotes)
        int modIdx = FindOperatorOutsideQuotes(expr, '%', requireDouble: false);
        if (modIdx >= 0)
        {
            var left = Eval(expr[..modIdx], env, errors);
            var right = Eval(expr[(modIdx + 1)..], env, errors);
            long ln = left is long ll ? ll : left is int ii ? ii : 0;
            long rn = right is long rl ? rl : right is int ri ? ri : 1;
            return (object)(rn != 0 ? ln % rn : 0);
        }

        // String concatenation: expr + expr (outside quotes)
        int plusIdx = FindOperatorOutsideQuotes(expr, '+', requireDouble: false);
        if (plusIdx >= 0)
        {
            var left = Eval(expr[..plusIdx], env, errors);
            var right = Eval(expr[(plusIdx + 1)..], env, errors);
            return (object)(Stringify(left) + Stringify(right));
        }

        // General subtraction: X - Y (bracket/paren-aware, finds top-level -)
        int subIdx = FindSubOutside(expr);
        if (subIdx >= 0)
        {
            var lv = Eval(expr[..subIdx].TrimEnd(), env, errors);
            var rv = Eval(expr[(subIdx + 1)..].TrimStart(), env, errors);
            long ln = lv is long ll ? ll : lv is int ii ? ii : 0;
            long rn = rv is long rl ? rl : rv is int ri ? ri : 0;
            return (object)(ln - rn);
        }

        // Filter: expr | trim
        var filterM = RegexGenerated.JinjaExprTrim.Match(expr);
        if (filterM.Success)
        {
            var v = Eval(filterM.Groups[1].Value, env, errors);
            return (object)(Stringify(v).Trim());
        }

        // Filter: expr | length
        var lengthM = RegexGenerated.JinjaExprLength.Match(expr);
        if (lengthM.Success)
        {
            var v = Eval(lengthM.Groups[1].Value, env, errors);
            return (object)(v switch
            {
                List<JinjaDict> list => (long)list.Count,
                string s => (long)s.Length,
                _ => 0L
            });
        }

        // Filter: expr | tojson  or  expr | tojson(indent=N)
        var tojsonM = RegexGenerated.JinjaExprToJson.Match(expr);
        if (tojsonM.Success)
        {
            var v = Eval(tojsonM.Groups[1].Value, env, errors);
            if (v is null) return (object)"null";
            if (v is string sv) return (object)System.Text.Json.JsonSerializer.Serialize(sv);
            if (v is bool bv) return (object)(bv ? "true" : "false");
            if (v is long lv) return (object)lv.ToString();
            if (v is int iv) return (object)iv.ToString();
            if (v is JinjaDict dict) return (object)DictToJson(dict);
            if (v is List<JinjaDict> list) return (object)ListToJson(list);
            return (object)Stringify(v);
        }

        // content.split('delim')[-1]  — last segment
        var splitM = RegexGenerated.JinjaSplitDelim.Match(expr);// Regex.Match(expr,@"^(\w+)\.split\('([^']*)'\)\[(-?\d+)\]$");
        if (splitM.Success)
        {
            var src = Stringify(Eval(splitM.Groups[1].Value, env, errors));
            string sep = splitM.Groups[2].Value;
            int idx = int.Parse(splitM.Groups[3].Value);
            var parts = src.Split(sep);
            int real = idx < 0 ? parts.Length + idx : idx;
            return (object)(real >= 0 && real < parts.Length ? parts[real] : "");
        }

        // String literal (single or double quotes)
        if ((expr.StartsWith('\'') && expr.EndsWith('\'')) ||
            (expr.StartsWith('"') && expr.EndsWith('"')))
        {
            string lit = expr[1..^1];
            // Unescape \\n -> newline etc.
            lit = lit.Replace("\\n", "\n").Replace("\\t", "\t");
            return (object)lit;
        }

        // Boolean literals
        if (expr == "true") return (object)true;
        if (expr == "false") return (object)false;
        if (expr == "none" || expr == "None") return null;

        // Integer literal
        if (long.TryParse(expr, out long lval)) return (object)lval;

        // obj[key] — find the LAST [...] at bracket depth 0 (right-to-left scan).
        // Must come before the dot-access check so messages[0]['role'] resolves
        // as (messages[0])['role'] rather than being mis-split by a regex.
        var (bracketStart, bracketEnd) = FindLastBracketPair(expr);
        if (bracketStart >= 0)
        {
            var obj = Eval(expr[..bracketStart], env, errors);
            string rawKey = expr[(bracketStart + 1)..bracketEnd];

            // Jinja slice syntax: [start:end], [:end], [start:]
            int colonIdx = FindColonOutsideQuotes(rawKey);
            if (colonIdx >= 0)
            {
                int? sliceStart = null, sliceEnd = null;
                if (colonIdx > 0)
                {
                    string startExpr = rawKey[..colonIdx].Trim();
                    if (startExpr.Length > 0)
                    {
                        var sv = Eval(startExpr, env, errors);
                        sliceStart = sv is long l ? (int)l : sv is int i ? i : null;
                    }
                }
                if (colonIdx + 1 < rawKey.Length)
                {
                    string endExpr = rawKey[(colonIdx + 1)..].Trim();
                    if (endExpr.Length > 0)
                    {
                        var ev = Eval(endExpr, env, errors);
                        sliceEnd = ev is long l ? (int)l : ev is int i ? i : null;
                    }
                }
                switch (obj)
                {
                    case List<JinjaDict> list:
                        {
                            int sIdx = sliceStart ?? 0;
                            int eIdx = sliceEnd ?? list.Count;
                            sIdx = sIdx < 0 ? list.Count + sIdx : sIdx;
                            eIdx = eIdx < 0 ? list.Count + eIdx : eIdx;
                            sIdx = Math.Clamp(sIdx, 0, list.Count);
                            eIdx = Math.Clamp(eIdx, 0, list.Count);
                            return sIdx >= eIdx ? [] : list[sIdx..eIdx];
                        }
                    case string s:
                        {
                            int sIdx = sliceStart ?? 0;
                            int eIdx = sliceEnd ?? s.Length;
                            sIdx = Math.Clamp(sIdx, 0, s.Length);
                            eIdx = Math.Clamp(eIdx, 0, s.Length);
                            return sIdx >= eIdx ? "" : s[sIdx..eIdx];
                        }
                    default: return null;
                }
            }

            var key = Eval(rawKey, env, errors);
            var idxVal = AccessIndex(obj, key);

            // Handle .field suffix after bracket access, e.g. messages[0].role
            if (idxVal != null && bracketEnd + 1 < expr.Length && expr[bracketEnd + 1] == '.')
                return AccessField(idxVal, expr[(bracketEnd + 2)..]);

            return idxVal;
        }

        // obj.field  (dot access, handles loop.first, ns.field etc.)
        // Also matches expressions like messages[0].role when no bracket pair
        // is found (e.g. after recursive Eval consumes the bracket part).
        var dotM = RegexGenerated.JinjaObjDotField.Match(expr);//  Regex.Match(expr, @"^(\w+)\.(\w+)$");
        if (dotM.Success)
        {
            var obj = env.Get(dotM.Groups[1].Value);
            string fld = dotM.Groups[2].Value;
            return AccessField(obj, fld);
        }

        // arithmetic: index0 +/- N inside bracket is handled by EvalIndex
        var arithM = RegexGenerated.JinjaPlusMinusN.Match(expr);// Regex.Match(expr, @"^(.+?)\s*([+\-])\s*(\d+)$");
        if (arithM.Success)
        {
            var lv = Eval(arithM.Groups[1].Value, env, errors);
            long rn = long.Parse(arithM.Groups[3].Value);
            long ln = lv is long ll ? ll : lv is int ii ? ii : 0;
            return (object)(arithM.Groups[2].Value == "+" ? ln + rn : ln - rn);
        }

        // Inline conditional: X if Y else Z / X if Y (lowest precedence)
        int ifPos = FindIfElseOutside(expr, out int elsePos);
        if (ifPos >= 0)
        {
            var trueVal = Eval(expr[..ifPos].TrimEnd(), env, errors);
            if (elsePos >= 0)
            {
                var cond = Eval(expr[(ifPos + 3)..elsePos].Trim(), env, errors);
                var falseVal = Eval(expr[(elsePos + 5)..].TrimStart(), env, errors);
                return IsTruthy(cond) ? trueVal : falseVal;
            }
            else
            {
                var cond = Eval(expr[(ifPos + 3)..].TrimStart(), env, errors);
                return IsTruthy(cond) ? trueVal : "";
            }
        }

        // X else Y  (default operator: if X is truthy return X, else return Y)
        // Must come after inline conditional so X if Y else Z is already consumed.
        {
            int elseOnlyPos = FindElseOutside(expr);
            if (elseOnlyPos >= 0)
            {
                var left = Eval(expr[..elseOnlyPos].TrimEnd(), env, errors);
                var right = Eval(expr[(elseOnlyPos + 5)..].TrimStart(), env, errors);
                return IsTruthy(left) ? left : right;
            }
        }

        // Builtin function calls: name(arg)
        var funcM = RegexGenerated.JinjaFuncCall.Match(expr);
        if (funcM.Success)
        {
            string funcName = funcM.Groups[1].Value;
            string argExpr = funcM.Groups[2].Value;
            if (funcName == "strftime_now")
            {
                // Format string like "%d %b %Y" or empty for default
                string fmt = argExpr.Trim('"', '\'', ' ');
                try { return (object)(fmt.Length > 0 ? DateTime.Now.ToString(fmt) : DateTime.Now.ToString("dd MMM yyyy")); }
                catch { return (object)DateTime.Now.ToString("dd MMM yyyy"); }
            }
            if (funcName == "raise_exception")
            {
                // Jinja2 built-in: aborts rendering with the given message.
                // We silently return empty string so the template continues.
                return (object)"";
            }
            _ = argExpr; // suppress unused-variable warning
            errors?.Add($"JINJA ERROR: unknown function '{funcName}'");
            return null;
        }

        // Plain identifier
        var result = env.Get(expr);
        if (result is null && !env.ContainsKey(expr))
        {
            string msg = $"JINJA ERROR: variable '{expr}' not found in template env";
            errors?.Add(msg);
            SanityChecks.WriteLine($"JinjaTemplateFormatter: variable '{expr}' not found in template env");
        }
        return result;
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

    private static object? AccessIndex(object? obj, object? key)
    {
        if (obj is JinjaDict dict)
        {
            string k = Stringify(key);
            return dict.TryGetValue(k, out var v) ? v : null;
        }
        if (obj is List<JinjaDict> list)
        {
            long idx = key is long l ? l : key is int i ? i : 0;
            int real = idx < 0 ? list.Count + (int)idx : (int)idx;
            return (real >= 0 && real < list.Count) ? list[real] : null;
        }
        return null;
    }

    private static int FindColonOutsideQuotes(string s)
    {
        bool inSingle = false, inDouble = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble && c == ':')
                return i;
        }
        return -1;
    }

    private static object? AccessField(object? obj, string field)
    {
        if (obj is JinjaDict dict)
            return dict.TryGetValue(field, out var v) ? v : null;
        if (obj is JinjaNamespace ns)
            return ns.Get(field);
        return null;
    }

    /// <summary>
    /// Scans right-to-left to find the last <c>[…]</c> pair at bracket depth 0,
    /// respecting string quotes. Returns <c>(openIdx, closeIdx)</c> where
    /// <c>closeIdx</c> points to the matching <c>]</c>, which may not be the
    /// final character (e.g., <c>messages[0].role</c> has trailing <c>.role</c>).
    /// Returns <c>(-1, -1)</c> when no bracket pair is found.
    /// </summary>
    private static (int Open, int Close) FindLastBracketPair(string s)
    {
        if (s.Length == 0) return (-1, -1);
        int closeBracket = s.LastIndexOf(']');
        if (closeBracket < 0) return (-1, -1);
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = closeBracket; i >= 0; i--)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c == ']') depth++;
                else if (c == '[')
                {
                    depth--;
                    if (depth == 0) return (i, closeBracket);
                }
            }
        }
        return (-1, -1);
    }


    /// When <paramref name="requireDouble"/> is true, matches only <c>==</c> (not <c>=</c>
    /// inside <c>!=</c> or standalone).
    /// </summary>
    private static int FindOperatorOutsideQuotes(string s, char op, bool requireDouble = false)
    {
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == op)
                {
                    if (requireDouble)
                    {
                        if (i + 1 < s.Length && s[i + 1] == '=' && (i == 0 || s[i - 1] != '!'))
                            return i;
                    }
                    else
                    {
                        return i;
                    }
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds <c>!=</c> outside string quotes and brackets.
    /// Returns the index of <c>!</c>, or -1 if not found.
    /// </summary>
    private static int FindNeqOutside(string s)
    {
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 0; i < s.Length - 1; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == '!' && s[i + 1] == '=')
                    return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds binary <c>-</c> (subtraction) outside string quotes, parens, and
    /// brackets. Returns the index of <c>-</c>, or -1 if not found.
    /// Skips unary negation (at position 0) and <c>--</c> / <c>-=</c>.
    /// </summary>
    private static int FindSubOutside(string s)
    {
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == '-' && i > 0)
                {
                    if (i + 1 < s.Length && (s[i + 1] is '-' or '=')) continue;
                    return i;
                }
            }
        }
        return -1;
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
    /// Finds <c>if</c> and optionally <c>else</c> in an inline conditional
    /// expression <c>X if Y else Z</c>, outside string quotes and brackets.
    /// Returns the position of <c>if</c> (the <c>i</c> in <c>if</c>),
    /// and sets <paramref name="elsePos"/> to the position of <c>else</c>
    /// (or -1 if not present).
    /// </summary>
    private static int FindIfElseOutside(string s, out int elsePos)
    {
        elsePos = -1;
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 1; i < s.Length - 3; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == ' ' && s[i - 1] != ' ' && s[i + 1] == 'i' && s[i + 2] == 'f' && s[i + 3] == ' ')
                {
                    int ifPos = i + 1;
                    for (int j = ifPos + 4; j < s.Length - 5; j++)
                    {
                        char d = s[j];
                        if (d == '\'' && !inDouble) inSingle = !inSingle;
                        else if (d == '"' && !inSingle) inDouble = !inDouble;
                        else if (!inSingle && !inDouble)
                        {
                            if (d is '(' or '[') depth++;
                            else if (d is ')' or ']') depth--;
                            else if (depth == 0 && d == ' ' && s[j - 1] != ' ' && s[j + 1] == 'e' && s[j + 2] == 'l' && s[j + 3] == 's' && s[j + 4] == 'e' && s[j + 5] == ' ')
                            {
                                elsePos = j + 1;
                                break;
                            }
                        }
                    }
                    return ifPos;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds a comparison operator (<c>&lt;</c>, <c>&gt;</c>, <c>&lt;=</c>, <c>&gt;=</c>)
    /// outside string quotes and brackets. Returns the index and the matched operator.
    /// Skips <c>!=</c> (already handled by <see cref="FindNeqOutside"/>).
    /// </summary>
    private static int FindCmpOutside(string s, out string op)
    {
        op = "";
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && (c == '<' || c == '>'))
                {
                    // != is handled elsewhere — skip it
                    if (c == '>' && i > 0 && s[i - 1] == '!') continue;
                    if (i + 1 < s.Length && s[i + 1] == '=')
                    {
                        op = c == '<' ? "<=" : ">=";
                        return i;
                    }
                    op = c == '<' ? "<" : ">";
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds standalone <c>else</c> (default operator) outside string quotes
    /// and brackets. Returns the position of <c>else</c> (the <c>e</c>), or -1.
    /// Must not be after <c>if</c> (that case is already consumed by
    /// <see cref="FindIfElseOutside"/>).
    /// </summary>
    private static int FindElseOutside(string s)
    {
        bool inSingle = false, inDouble = false;
        int depth = 0;
        for (int i = 1; i < s.Length - 5; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble)
            {
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && c == ' ' && s[i - 1] != ' ' && s[i + 1] == 'e' && s[i + 2] == 'l' && s[i + 3] == 's' && s[i + 4] == 'e' && s[i + 5] == ' ')
                    return i + 1;
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
