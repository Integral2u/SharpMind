namespace SandBox.RunModels
{
    public class AllQwenModels
    {
        //All Qwen2-0.5B work but can produce giberish and fail to emmit eos token. they require "Q: What is the capital of France?\nA:" format and string stops. e.g. "\n\n", "\nQ:"
        //All Qwen2-0.5B-instruct work
        public static readonly string[] Models =
        [
            /*Prompt:What is the capital of france
            Response:the capital of the of the franc is 10. The 10.0.
            The capital in dollars, in dollars, and in dollars are respectively.
            The dollar is the dollar of the dollar is equal to to equal to be equal. The 00.0.
            The capital dollars, in dollars, and in dollars are respectively.
            The capital of the dollar is equal to be to be equal to equal. The 100.
            The capital of the dollars, and in dollars are respectively.
            The capital dollars, and dollar is respectively.
            Tokens per second: 1.22  TTFT: 23.470s*/
            "Qwen2-0.5B.Q2_K", 

            /*Prompt:What is the capital of france
            Response:the capital of the franc is 100.5%of it is 2.5%34876998%
            The capital of the franc in dollars, in 1/0.001/25%25%3447689989898767. The capital in dollars, in 0.01/225%34%54%67398988987. The capital in dollars, in 0.01/25/25%4%4673363989898. The capital in dollars, in 0.0125/25%4%4/77366369898. The capital in dollars, in 0.0105/25%4%4%77789366898. The capital in dollars, in 1.00025/52/4%4%7739896689. The capital in dollars, in 1.00025/52/
            Tokens per second: 0.89  TTFT: 25.688s*/
            //"Qwen2-0.5B.Q3_K_L",

            /*Prompt:What is the capital of france
            Response:the capital of the franc is 10. 20.3%.
            Thecapitalization of the capital in this in this is 1/3/4%.
            Thecapitalizationofthe capital is 10.0%.
            Thecapitalization of the capital in the in this is 10.3%.2.
            Thecapitalizationofthe capital is 1/30%.
            Thecapitalization of the capital in this in this is 10.%
            Thecapitalization of the capital in this it is 1%.20%.
            Thecapitalization of the capital in this in it is 10.%
            Thecapitalization of the capital in this it is 1%. Thecapitalizationofthecapitalinthisit%.
            TheCapitalizationof Capital Capital In This It Is 0.0%
            Thecapital in this in it is 1%. The capitalization of the capital in this%.
            Capitalizationof Capital In This It Is 0.0%
            Thecapitalization of the in it is 1%. Thecapitalization of capital in this%.
            Capital In This Capital Is The Capital It Is 0.0%
            Thecapitalization of the capital in this is 1%. Thecapitalizationof capital in
            Tokens per second: 1.55  TTFT: 13.790s*/
            //"Qwen2-0.5B.Q3_K_M",

            /*Prompt:What is the capital of france
            Response:The capital of the of the French is 1. The capital is 10.
            The capital of the French is 0.
            The Capital in and in France are both zero.
            Tokens per second: 0.95  TTFT: 23.167s*/
            //"Qwen2-0.5B.Q3_K_S",

            /*Prompt:What is the capital of france
            Response:the capital franc is the amount of money that can be used to buy things. The amount that can be called the value or price.
            The value of money in dollars, expressed as a number, is called its unit. The unit has one thing to it's own value. It can be measured by dividing the amount that money can buy by itself.
            The amount of money, expressed as a number, is called its denomination. The denomination has one thing to it's own value. It can be measured by dividing the amount that money buy by itself.
            The amount of money, expressed as a number, is called its denomination. The denomination has one thing to be divided by itself to get the amount that can buy.
            The amount of money in dollars, expressed as a number, is called its value. The value has one thing to be divided by itself to get buy.
            The amount that can be bought in dollars, expressed as a number, is called its price. The value of money can be measured by dividing the amount buy it's value by itself.
            The amount that can be bought in dollars expressed as a number, is called its unit. The value of money can be measured by dividing the buy it's own value.
            The amount that can be bought in dollars expressed as a
            Tokens per second: 1.56  TTFT: 18.366s*/
            //"Qwen2-0.5B.Q5_1",

            /*Prompt:What is the capital of france
            Response:what is the capital of the francs in 19020345678982345671892367894501 1 0 0 2345698734567898236279451011 0 0 2345698734567898986237452011 0 0 1345629734567898989637452011 0 0 1 3452679456783898923745620110 0 1 3452679456783898923745620110 0 1 3452679456783898923745620110 0 1 3452679
            Tokens per second: 2.47  TTFT: 10.933s*/
            //"Qwen2-0.5B.Q6_K",
            /*Prompt:What is the capital of france
            Response:the capital of the franc is 100.5%of it is 2.5% and the rest 4%. The total amount that you have in your money in dollars is 1,000.5% of it is 25% and the rest of it is 33. The total amount that you have in dollars is 1,00.5%. The capitalization of your money in dollars will be 1, and the rest will be 2%.
            The amount that you have in dollars is0.5%. The total amount of your money in dollars will be 1, and the rest will be 2%.
            The amount that you have in dollars is0.5%. The total amount that you have in dollars will be 1, and the rest will be 2%.
            The amount that you have in dollars is0.5%. The total amount that you have in dollars will be 1, and the rest will be 2%.
            The amount that you have in dollars is0.5%. The total amount that you have in dollars will be 1, and the rest will be 2%.
            The amount that you have in dollars is0.5%. The total amount that
            Tokens per second: 5.33  TTFT: 5.146s*/
            //"Qwen2-0.5B.Q8_0",

            /*Prompt:What is the capital of France?
            Response:Paris.
            Tokens per second: 0.45  TTFT: 30.810s*/
            //"qwen2-0.5b-instruct-q2_k",
            /*Prompt:What is the capital of France?
            Response:Paris

            The capital of France is Paris. The French government's official language and cultural identity are both French, which makes it the most widely spoken language in Europe and North America.

            Tokens per second: 1.15  TTFT: 26.107s*/
            "qwen2-0_5b-instruct-q4_k_m",       //Prompt:Hello Response:Hello! How can I assist you today?            
            /*Prompt:What is the capital of France?
            Response:Paris.
            Tokens per second: 4.46  TTFT: 4.415s*/
            "qwen2-0_5b-instruct-q8_0",         //Prompt:Hello Response:Hello! How can I assist you today?
            /*Prompt:What is the capital of France?
            Response:Paris.

            Tokens per second: 0.66  TTFT: 40.585s*/
            //"qwen2-0_5b-instruct-fp16",         //Prompt:Hello Response:Hello! How can I assist you today?  
            /*Prompt:What is the capital of France?
            Response:The
            France is the capitale of France, and Francia in Francaland.

            The

            France's capitalele.


            Tokens per second: 1.48  TTFT: 43.148s*/
            "qwen2.5-1.5b-instruct-q8_0",       //Prompt:Hello Response:Hello! How can I help you today?
            /*Prompt:What is the capital of France?
            Response:France's Capital

            Tokens per second: 0.18  TTFT: 201.302s*/
            //"Qwen2.5-1.5B-Instruct-f16",        //Prompt:Hello Response:I am sorry for my mistake I did not understand your message correctly. Could you please rephrase the question or statement that you
            /*Prompt:What is the capital of France?
            Response:The capital city of France is Paris.

            Paris, located in the Ile-de-France region, has been the seat of French government since 1285.

            The city's rich history and cultural significance make it one of the most visited cities in the world, attracting millions each year with its iconic landmarks such as Eiffel Tower, Louvre Museum, Notre-Dame Cathedral and many more.

            Paris has also been home to numerous influential figures in art, literature and politics throughout history including Claude Monet, Victor Hugo, Napoleon Bonaparte and many more.

            In conclusion Paris remains the capital city of France to this day with its rich cultural heritage and historical significance making it one of the most visited cities in the world.

            Tokens per second: 0.71  TTFT: 100.861s*/
            //"qwen2.5-coder-3b-instruct-q8_0",   //Prompt:Hello Response: Hello! How can I assist you today?
            /*Prompt:What is the capital of France?
            Response:The capital of France is Paris.

            Paris, located in the north-central part of France, has been the country's political and cultural center since ancient times.

            Paris was founded as a Roman settlement around 50 BC, and became an important city in Western Europe during the Middle Ages.

            In modern times, Paris has developed into one of the world's most famous cities for its rich history and culture, as well as its stunning architecture and beautiful scenery.

            Paris is also known for its high-quality education system, with many prestigious universities located in Paris.

            In summary, Paris has been the capital of France since ancient times and has developed into one of the world's most famous cities for its rich history and culture, as well as its stunning architecture and beautiful scenery.

            Tokens per second: 0.13  TTFT: 284.739s*/
            //"qwen2.5-coder-3b-instruct-q4_k_m",
            /*Prompt:What is the capital of France?
            Response:The capital city of France is Paris, located on the I in the north-western part of Europe.

            Tokens per second: 0.09  TTFT: 414.370s*/
            //"qwen2.5-coder-3b-instruct-q2_k",

            /*Prompt:What is the capital of France?
            Response:



            Okay, let's see. The user is asking for the capital of France. But I know that in any country, their capital can be a city or a country itself.

            So, since France's capitals are not known yet. So maybe I should explain that the actual answer for this question is "France has no capital in its own country.

            But since you're asking, then maybe I can explain that. But also, since it's a bit of a small question.

            So the answer should be "France has no capital in itself. So for this question, I can explain that. But since it's a bit of small question.

            So the answer should be "France has no capital in itself. So for this question, I can explain that."

            **

            **The actual capital of France is not known yet. Since we are talking about it's a bit small question. So for this, answer should be "France has no capital in itself."**

            ****

            The actual answer is: **France has no capital in itself. So for this question, I can explain that.****

            \boxed{France} has no capital in itself.

            ** **

            **

            The actual answer is: **France has no capital in itself..

            Tokens per second: 0.44  TTFT: 49.624s*/
            //"Qwen3-0.6B-Q2_K",                  //Prompt:Hello Response:?\nOkay, so I need to start with the user's message. 
           // "Qwen3-0.6B-Q3_K_M",                //Prompt:Hello Response:\nOkay, the user is asking for help with a problem. But
            //"Qwen3-0.6B-Q4_0",                  //Prompt:Hello Response:Okay, the user asked me to respond in a specific way.
            //"Qwen3-0.6B-Q4_1",                  //Prompt:Hello Response:Okay, the user just said "Hello and I need to respond    
            //"Qwen3-0.6B-Q4_K_M",                //Prompt:Hello Response:\nOkay, the user just said "Hello," so I need to 
            //"Qwen3-0.6B-Q5_K_M",                //Prompt:Hello Response:\nOkay, the user just said "Hello," so I need to                                   
            //"Qwen3-0.6B-Q6_K",                  //Prompt:Hello Response:\nOkay, the user just said "Hello," so I need to             
            "Qwen3-0.6B-Q8_0",                  //Prompt:Hello Response:\nOkay, the user just said "Hello," so I need to

            //"DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", //Prompt:Hello Response:Hi! Welcome to Brain. I'm Trying To Teach Help '>>\n\nAlright, so I need help with this. The equation
            //"DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",   //Prompt:Hello Response:Hi, thank you for asking your question. I'm just going through my memory again. Times ago, I have this busy

];

private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
public static async Task RunAsync(string prompt, bool diag)
{

if (diag)
{
//GeneratorDiagnostics.DumpTopLogits = true;
await DiagnosticModelRunner.RunAsync(prompt, Models);
}
else await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
}  
}
}
