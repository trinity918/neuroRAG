"""Author + emit the bundled multilingual neuroscience benchmark.

Running this script (re)creates two artifacts referenced by configs/default.yaml:

    data/corpus/neuro_corpus.jsonl   -- the multilingual passage corpus
    data/eval/queries.jsonl          -- evaluation queries with relevance judgments

The content is hand-authored here (rather than shipped as raw JSONL) so that
(a) the multilingual text is guaranteed to serialize as valid UTF-8 JSON, and
(b) the benchmark is fully reproducible from source. Parallel English/Hindi
passages on shared topics are intentional: they are what make the
cross-lingual retrieval evaluation meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Corpus. Each passage is a short, concept-dense abstract. `topic` is a coarse
# label used only to keep the relevance judgments below consistent.
# --------------------------------------------------------------------------- #
DOCS: list[dict] = [
    # ---- English ----
    dict(id="en-n400", lang="en", topic="n400", year=2019,
         title="The N400 as an index of semantic processing",
         source="Cognitive Neuroscience Review",
         text="The N400 is a negative event-related potential peaking around 400 ms after a word. "
              "Its amplitude grows when a word is semantically incongruent with its context, so the "
              "N400 is widely used as an electrophysiological index of semantic memory and language "
              "comprehension. It is measured with EEG by averaging time-locked epochs across trials."),
    dict(id="en-p300", lang="en", topic="p300", year=2018,
         title="P300 in the oddball paradigm and the P300 speller",
         source="Journal of BCI",
         text="The P300 is a positive ERP appearing about 300 ms after a rare, task-relevant stimulus in "
              "the oddball paradigm. Because it reflects attention and stimulus evaluation, the P300 "
              "drives the P300 speller, a brain-computer interface that lets users select letters from a "
              "flashing matrix without any muscle movement."),
    dict(id="en-n170", lang="en", topic="n170", year=2017,
         title="The face-selective N170 component",
         source="Vision and Cognition",
         text="The N170 is a negative ERP peaking near 170 ms over occipito-temporal electrodes. It is "
              "substantially larger for faces than for other objects, making the N170 a marker of early "
              "face processing in the visual cortex. It is typically recorded with EEG."),
    dict(id="en-mmn", lang="en", topic="mmn", year=2016,
         title="Mismatch negativity and pre-attentive deviance detection",
         source="Auditory Neuroscience",
         text="The mismatch negativity (MMN) is elicited when a deviant sound violates a regularity "
              "established by standard sounds in the auditory oddball. Generated in temporal cortex, the "
              "MMN reflects pre-attentive change detection and is reduced in schizophrenia, where it "
              "serves as a candidate biomarker."),
    dict(id="en-p600", lang="en", topic="p600", year=2015,
         title="The P600 and syntactic reanalysis",
         source="Psycholinguistics Quarterly",
         text="The P600 is a late positive ERP around 600 ms that increases for grammatical violations and "
              "garden-path sentences. It is interpreted as an index of syntactic reanalysis and "
              "integration during language processing, complementing the semantically driven N400."),
    dict(id="en-ern", lang="en", topic="ern", year=2014,
         title="Error-related negativity and performance monitoring",
         source="Cognitive Control Studies",
         text="The error-related negativity (ERN) is a fronto-central negativity that appears within 100 ms "
              "of an erroneous response. Generated near the frontal cortex, it reflects performance "
              "monitoring and is enlarged in anxiety, linking cognitive control to affective state."),
    dict(id="en-alpha", lang="en", topic="alpha", year=2019,
         title="Posterior alpha rhythm and attention",
         source="EEG Rhythms Handbook",
         text="The alpha rhythm is an 8 to 12 Hz oscillation that dominates occipital EEG when the eyes are "
              "closed and the participant is relaxed. Alpha power decreases with visual attention and "
              "increases during internally directed states, making it a robust marker of cortical "
              "idling and attentional gating."),
    dict(id="en-theta", lang="en", topic="theta", year=2020,
         title="Theta oscillations, memory, and the hippocampus",
         source="Memory Systems Journal",
         text="Theta oscillations between 4 and 8 Hz are generated in part by the hippocampus and increase "
              "during working memory load and spatial navigation. Frontal midline theta scales with task "
              "difficulty, supporting the view that theta coordinates memory encoding and cognitive "
              "control."),
    dict(id="en-gamma", lang="en", topic="gamma", year=2018,
         title="Gamma-band activity and feature binding",
         source="Oscillations and Cognition",
         text="Gamma-band activity above 30 Hz has been linked to attention and to the binding of features "
              "into coherent percepts. Gamma power rises with selective attention, and abnormal gamma "
              "synchronization is reported in schizophrenia."),
    dict(id="en-delta", lang="en", topic="delta", year=2017,
         title="Delta waves in deep sleep",
         source="Sleep EEG Reviews",
         text="Delta waves are high-amplitude oscillations below 4 Hz that dominate slow-wave sleep. The "
              "amount of delta activity is the principal marker used to identify deep NREM stages during "
              "sleep staging from EEG."),
    dict(id="en-mu", lang="en", topic="mu", year=2019,
         title="The mu rhythm and motor imagery",
         source="Sensorimotor Neuroscience",
         text="The mu rhythm is a sensorimotor oscillation near 10 Hz that is suppressed when a person "
              "moves or imagines moving. This event-related desynchronization during motor imagery is a "
              "core control signal for non-invasive brain-computer interfaces."),
    dict(id="en-ica", lang="en", topic="ica", year=2016,
         title="Independent component analysis for EEG artifact removal",
         source="Signal Processing for EEG",
         text="Independent component analysis (ICA) separates EEG into maximally independent sources. "
              "Because eye blinks and muscle activity project onto distinct components, ICA is a standard "
              "tool for artifact rejection, letting analysts remove contamination while preserving neural "
              "signals."),
    dict(id="en-erp-method", lang="en", topic="erp_method", year=2015,
         title="Building clean ERPs by trial averaging",
         source="ERP Methods Primer",
         text="An event-related potential is obtained by averaging many EEG epochs that are time-locked to "
              "an event. Averaging cancels random background activity so that the evoked components emerge. "
              "Careful filtering, referencing, and epoching precede averaging to control noise."),
    dict(id="en-tf", lang="en", topic="time_frequency", year=2018,
         title="Time-frequency analysis with wavelets",
         source="Spectral Methods Review",
         text="Time-frequency analysis decomposes EEG into power across time and frequency, revealing "
              "transient bursts that averaging hides. Morlet wavelet convolution is a common approach and "
              "is used to quantify event-related synchronization and desynchronization in specific bands."),
    dict(id="en-source", lang="en", topic="source", year=2017,
         title="Source localization and the EEG inverse problem",
         source="Neuroimaging Methods",
         text="Source localization estimates the cortical generators of scalp EEG by solving an ill-posed "
              "inverse problem. Distributed methods such as sLORETA and beamformers use a head model to "
              "map surface potentials back to brain regions."),
    dict(id="en-connectivity", lang="en", topic="connectivity", year=2020,
         title="Functional connectivity and network breakdown in Alzheimer's disease",
         source="Clinical Neurophysiology",
         text="Functional connectivity measures statistical coupling between brain regions, for example "
              "with coherence or phase-locking value. In Alzheimer's disease, EEG connectivity in the "
              "alpha band is disrupted, reflecting a breakdown of large-scale networks that tracks "
              "cognitive decline."),
    dict(id="en-csp", lang="en", topic="csp", year=2016,
         title="Common spatial patterns for motor-imagery BCI",
         source="Machine Learning for EEG",
         text="Common spatial patterns (CSP) learn spatial filters that maximize the variance difference "
              "between two motor-imagery classes. The resulting features feed a classifier and remain a "
              "strong baseline for decoding left- versus right-hand imagery in brain-computer interfaces."),
    dict(id="en-ssvep", lang="en", topic="ssvep", year=2018,
         title="SSVEP-based brain-computer interfaces",
         source="Journal of BCI",
         text="A steady-state visual evoked potential (SSVEP) is an oscillatory EEG response entrained to a "
              "flickering stimulus. By tagging targets with different flicker frequencies, an SSVEP "
              "brain-computer interface can achieve high information transfer rates with little training."),
    dict(id="en-epilepsy", lang="en", topic="epilepsy", year=2019,
         title="EEG in the diagnosis of epilepsy",
         source="Clinical Neurophysiology",
         text="EEG is central to diagnosing epilepsy because interictal spikes and sharp waves reveal "
              "cortical hyperexcitability between seizures. During a seizure the EEG shows rhythmic, highly "
              "synchronized discharges that help localize the epileptogenic zone."),
    dict(id="en-alzheimer", lang="en", topic="alzheimer", year=2020,
         title="EEG slowing in Alzheimer's disease",
         source="Dementia Research",
         text="Alzheimer's disease produces a characteristic slowing of the EEG, with increased delta and "
              "theta power and reduced alpha and beta activity. These spectral changes, together with "
              "reduced functional connectivity, are studied as low-cost markers of dementia progression."),
    dict(id="en-depression", lang="en", topic="depression", year=2018,
         title="Frontal alpha asymmetry in depression",
         source="Affective Neuroscience",
         text="Major depressive disorder has been associated with frontal alpha asymmetry, a relative "
              "difference in alpha power between the left and right frontal cortex. The measure is proposed "
              "as an EEG marker of approach and withdrawal motivation, though its reliability is debated."),
    dict(id="en-adhd", lang="en", topic="adhd", year=2017,
         title="The theta/beta ratio in ADHD",
         source="Developmental Neuroscience",
         text="An elevated theta-to-beta ratio in frontal EEG has been reported in attention deficit "
              "hyperactivity disorder. Although once proposed as a diagnostic marker and a neurofeedback "
              "target, its specificity across individuals is limited."),
    dict(id="en-sleep", lang="en", topic="sleep", year=2019,
         title="Sleep spindles and K-complexes in stage 2 sleep",
         source="Sleep EEG Reviews",
         text="Stage 2 non-REM sleep is defined by sleep spindles, brief 11 to 16 Hz bursts, and by "
              "K-complexes, large biphasic waveforms. Automated sleep staging systems detect these "
              "graphoelements in the EEG to classify sleep stages."),
    dict(id="en-neurofeedback", lang="en", topic="neurofeedback", year=2016,
         title="Neurofeedback and self-regulation of EEG",
         source="Applied Neuroscience",
         text="Neurofeedback trains people to self-regulate their own EEG by presenting real-time feedback "
              "of a target rhythm, such as the sensorimotor rhythm. It has been explored as a treatment "
              "for ADHD and epilepsy, although rigorous controlled evidence remains mixed."),
    dict(id="en-dl", lang="en", topic="deep_learning", year=2021,
         title="Deep learning for EEG decoding",
         source="Machine Learning for EEG",
         text="Convolutional and recurrent neural networks now decode EEG for motor imagery, sleep staging, "
              "and seizure detection. Compact architectures such as EEGNet learn spatial and temporal "
              "filters end to end, often outperforming hand-crafted features like common spatial patterns "
              "for brain-computer interfaces."),
    dict(id="en-preproc", lang="en", topic="preproc", year=2018,
         title="A standard EEG preprocessing pipeline",
         source="ERP Methods Primer",
         text="A typical EEG preprocessing pipeline applies a band-pass filter, re-references the channels, "
              "removes bad segments, and runs independent component analysis for artifact rejection before "
              "epoching the data. Consistent preprocessing is essential for reproducible ERP and "
              "time-frequency results."),

    # ---- Hindi (parallel topics) ----
    dict(id="hi-n400", lang="hi", topic="n400", year=2019,
         title="N400 और अर्थपूर्ण प्रसंस्करण का सूचकांक",
         source="संज्ञानात्मक तंत्रिका विज्ञान समीक्षा",
         text="N400 एक ऋणात्मक घटना-संबंधी विभव है जो किसी शब्द के लगभग 400 मिलीसेकंड बाद शिखर पर पहुँचता है। "
              "जब कोई शब्द अपने संदर्भ के साथ अर्थपूर्ण रूप से असंगत होता है तो इसका आयाम बढ़ जाता है, इसलिए N400 को "
              "अर्थपूर्ण स्मृति और भाषा बोध के विद्युत-शारीरिक सूचकांक के रूप में उपयोग किया जाता है। इसे EEG से "
              "एपॉक के औसतन द्वारा मापा जाता है।"),
    dict(id="hi-p300", lang="hi", topic="p300", year=2018,
         title="ऑडबॉल प्रतिमान में P300 और P300 स्पेलर",
         source="बीसीआई पत्रिका",
         text="P300 एक धनात्मक ERP है जो ऑडबॉल प्रतिमान में किसी दुर्लभ, कार्य-प्रासंगिक उद्दीपन के लगभग 300 "
              "मिलीसेकंड बाद प्रकट होता है। चूँकि यह ध्यान को दर्शाता है, इसलिए P300 स्पेलर नामक मस्तिष्क-कंप्यूटर "
              "इंटरफ़ेस इसका उपयोग करता है, जिससे उपयोगकर्ता बिना किसी मांसपेशीय गति के अक्षर चुन सकते हैं।"),
    dict(id="hi-alpha", lang="hi", topic="alpha", year=2019,
         title="पश्च अल्फा लय और ध्यान",
         source="ईईजी लय पुस्तिका",
         text="अल्फा लय 8 से 12 हर्ट्ज़ की एक दोलन है जो आँखें बंद और विश्राम की स्थिति में पश्चकपाल EEG पर हावी "
              "रहती है। दृश्य ध्यान के साथ अल्फा शक्ति घटती है, जिससे यह प्रांतस्था की निष्क्रियता और ध्यान नियंत्रण "
              "का एक विश्वसनीय संकेतक बन जाती है।"),
    dict(id="hi-theta", lang="hi", topic="theta", year=2020,
         title="थीटा दोलन, स्मृति और हिप्पोकैम्पस",
         source="स्मृति तंत्र पत्रिका",
         text="4 से 8 हर्ट्ज़ की थीटा दोलन आंशिक रूप से हिप्पोकैम्पस द्वारा उत्पन्न होती है और कार्यशील स्मृति भार "
              "तथा स्थानिक नेविगेशन के दौरान बढ़ती है। ललाट मध्यरेखा थीटा कार्य कठिनाई के साथ बढ़ती है, जो दर्शाता है "
              "कि थीटा स्मृति एन्कोडिंग का समन्वय करती है।"),
    dict(id="hi-delta", lang="hi", topic="delta", year=2017,
         title="गहरी नींद में डेल्टा तरंगें",
         source="नींद ईईजी समीक्षा",
         text="डेल्टा तरंगें 4 हर्ट्ज़ से कम की उच्च-आयाम दोलन हैं जो धीमी-तरंग नींद पर हावी रहती हैं। EEG से नींद "
              "अवस्था निर्धारण के दौरान गहरे NREM चरणों की पहचान के लिए डेल्टा गतिविधि की मात्रा मुख्य संकेतक है।"),
    dict(id="hi-mu", lang="hi", topic="mu", year=2019,
         title="म्यू लय और मोटर कल्पना",
         source="संवेदी-प्रेरक तंत्रिका विज्ञान",
         text="म्यू लय लगभग 10 हर्ट्ज़ की एक संवेदी-प्रेरक दोलन है जो व्यक्ति के हिलने या हिलने की कल्पना करने पर "
              "दब जाती है। मोटर कल्पना के दौरान यह घटना-संबंधी विसंकालिकीकरण गैर-आक्रामक मस्तिष्क-कंप्यूटर इंटरफ़ेस "
              "के लिए एक प्रमुख नियंत्रण संकेत है।"),
    dict(id="hi-ica", lang="hi", topic="ica", year=2016,
         title="EEG आर्टिफ़ैक्ट हटाने के लिए स्वतंत्र घटक विश्लेषण",
         source="ईईजी सिग्नल प्रसंस्करण",
         text="स्वतंत्र घटक विश्लेषण (ICA) EEG को अधिकतम स्वतंत्र स्रोतों में विभाजित करता है। चूँकि आँखों की "
              "पलक और मांसपेशीय गतिविधि अलग-अलग घटकों पर प्रक्षेपित होती है, इसलिए ICA आर्टिफ़ैक्ट हटाने का एक "
              "मानक उपकरण है जो तंत्रिका संकेतों को सुरक्षित रखते हुए संदूषण को हटाता है।"),
    dict(id="hi-erp-method", lang="hi", topic="erp_method", year=2015,
         title="ट्रायल औसतन द्वारा स्वच्छ ERP बनाना",
         source="ईआरपी विधि प्राइमर",
         text="घटना-संबंधी विभव कई EEG एपॉक के औसतन से प्राप्त होता है जो किसी घटना के साथ समय-बद्ध होते हैं। "
              "औसतन यादृच्छिक पृष्ठभूमि गतिविधि को रद्द कर देता है ताकि विकसित घटक उभर सकें। औसतन से पहले सावधानीपूर्वक "
              "फ़िल्टरिंग और एपॉकिंग की जाती है।"),
    dict(id="hi-epilepsy", lang="hi", topic="epilepsy", year=2019,
         title="मिर्गी के निदान में EEG",
         source="नैदानिक तंत्रिका-शरीरक्रिया विज्ञान",
         text="मिर्गी के निदान में EEG केंद्रीय है क्योंकि दौरों के बीच अंतराल में स्पाइक और तीक्ष्ण तरंगें प्रांतस्थीय "
              "अति-उत्तेजना को प्रकट करती हैं। दौरे के दौरान EEG लयबद्ध, अत्यधिक समकालिक विसर्जन दिखाता है जो "
              "मिर्गीजनक क्षेत्र का स्थानीयकरण करने में मदद करता है।"),
    dict(id="hi-alzheimer", lang="hi", topic="alzheimer", year=2020,
         title="अल्ज़ाइमर रोग में EEG की मंदता",
         source="मनोभ्रंश अनुसंधान",
         text="अल्ज़ाइमर रोग EEG की एक विशिष्ट मंदता उत्पन्न करता है, जिसमें डेल्टा और थीटा शक्ति बढ़ जाती है और "
              "अल्फा तथा बीटा गतिविधि घट जाती है। ये स्पेक्ट्रल परिवर्तन, घटी हुई कार्यात्मक संयोजकता के साथ, मनोभ्रंश "
              "की प्रगति के कम लागत वाले संकेतक के रूप में अध्ययन किए जाते हैं।"),
    dict(id="hi-depression", lang="hi", topic="depression", year=2018,
         title="अवसाद में ललाट अल्फा असममितता",
         source="भावात्मक तंत्रिका विज्ञान",
         text="गंभीर अवसादग्रस्तता विकार को ललाट अल्फा असममितता से जोड़ा गया है, जो बाएँ और दाएँ ललाट प्रांतस्था के "
              "बीच अल्फा शक्ति का सापेक्ष अंतर है। इस माप को अभिगमन और परिहार प्रेरणा के EEG संकेतक के रूप में "
              "प्रस्तावित किया गया है।"),
    dict(id="hi-adhd", lang="hi", topic="adhd", year=2017,
         title="एडीएचडी में थीटा/बीटा अनुपात",
         source="विकासात्मक तंत्रिका विज्ञान",
         text="ध्यान अभाव अतिसक्रियता विकार में ललाट EEG में बढ़ा हुआ थीटा-से-बीटा अनुपात बताया गया है। हालाँकि इसे "
              "कभी नैदानिक संकेतक और न्यूरोफ़ीडबैक लक्ष्य के रूप में प्रस्तावित किया गया था, व्यक्तियों में इसकी "
              "विशिष्टता सीमित है।"),
    dict(id="hi-neurofeedback", lang="hi", topic="neurofeedback", year=2016,
         title="न्यूरोफ़ीडबैक और EEG का स्व-नियमन",
         source="अनुप्रयुक्त तंत्रिका विज्ञान",
         text="न्यूरोफ़ीडबैक लोगों को किसी लक्ष्य लय, जैसे संवेदी-प्रेरक लय, की वास्तविक-समय प्रतिक्रिया दिखाकर अपने "
              "EEG का स्व-नियमन करना सिखाता है। इसे एडीएचडी और मिर्गी के उपचार के रूप में खोजा गया है, हालाँकि कठोर "
              "नियंत्रित प्रमाण मिश्रित हैं।"),
    dict(id="hi-dl", lang="hi", topic="deep_learning", year=2021,
         title="EEG डिकोडिंग के लिए गहन शिक्षण",
         source="ईईजी के लिए मशीन लर्निंग",
         text="कन्वोल्यूशनल और आवर्ती तंत्रिका नेटवर्क अब मोटर कल्पना, नींद अवस्था निर्धारण और दौरा पहचान के लिए EEG "
              "को डिकोड करते हैं। EEGNet जैसी सुसंहत संरचनाएँ स्थानिक और अस्थायी फ़िल्टर सीखती हैं, जो अक्सर "
              "मस्तिष्क-कंप्यूटर इंटरफ़ेस के लिए हस्त-निर्मित विशेषताओं से बेहतर प्रदर्शन करती हैं।"),

    # ---- Bengali ----
    dict(id="bn-n400", lang="bn", topic="n400", year=2019,
         title="N400 এবং অর্থগত প্রক্রিয়াকরণ",
         source="সংজ্ঞানমূলক স্নায়ুবিজ্ঞান",
         text="N400 হল একটি ঋণাত্মক ঘটনা-সম্পর্কিত বিভব (ERP) যা কোনও শব্দের প্রায় 400 মিলিসেকেন্ড পরে শীর্ষে "
              "পৌঁছায়। প্রসঙ্গের সঙ্গে অর্থগতভাবে অসঙ্গতিপূর্ণ শব্দে এর মাত্রা বৃদ্ধি পায়, তাই N400 কে অর্থগত স্মৃতি "
              "ও ভাষা বোঝার সূচক হিসেবে EEG দিয়ে পরিমাপ করা হয়।"),
    dict(id="bn-epilepsy", lang="bn", topic="epilepsy", year=2019,
         title="মৃগীরোগ নির্ণয়ে EEG",
         source="ক্লিনিকাল স্নায়ুবিজ্ঞান",
         text="মৃগীরোগ নির্ণয়ে EEG অত্যন্ত গুরুত্বপূর্ণ, কারণ খিঁচুনির মধ্যবর্তী সময়ে স্পাইক ও তীক্ষ্ণ তরঙ্গ "
              "কর্টেক্সের অতি-উত্তেজনা প্রকাশ করে। খিঁচুনির সময় EEG ছন্দবদ্ধ, অত্যন্ত সমকালীন নিঃসরণ দেখায়।"),

    # ---- Tamil ----
    dict(id="ta-bci", lang="ta", topic="mu", year=2019,
         title="மோட்டார் கற்பனை மூளை-கணினி இடைமுகம்",
         source="மூளை-கணினி இடைமுக இதழ்",
         text="மியூ ரிதம் (mu rhythm) என்பது சுமார் 10 Hz அதிர்வு ஆகும், இது ஒருவர் அசையும்போது அல்லது அசைவதை "
              "கற்பனை செய்யும்போது அடக்கப்படுகிறது. மோட்டார் கற்பனையின் போது ஏற்படும் இந்த மாற்றம் EEG அடிப்படையிலான "
              "மூளை-கணினி இடைமுகத்திற்கு (BCI) முக்கியக் கட்டுப்பாட்டு சமிக்ஞையாகும்."),
    dict(id="ta-alpha", lang="ta", topic="alpha", year=2019,
         title="ஆல்ஃபா ரிதமும் கவனமும்",
         source="EEG அலைகள்",
         text="ஆல்ஃபா அலை என்பது 8 முதல் 12 Hz வரையிலான அதிர்வு ஆகும், இது கண்களை மூடிய ஓய்வு நிலையில் "
              "பிடபகுதி EEG-யில் ஆதிக்கம் செலுத்துகிறது. காட்சிக் கவனத்துடன் ஆல்ஃபா சக்தி குறைகிறது."),
]

# --------------------------------------------------------------------------- #
# Evaluation queries with relevance judgments.
#   type: factoid | cross-lingual | multi-hop | global
#   relevant: doc ids that satisfy the information need (binary relevance)
#   answer: short reference answer used by the RAGAS-style proxy metrics
# --------------------------------------------------------------------------- #
QUERIES: list[dict] = [
    dict(id="q01", lang="en", type="factoid",
         query="What does the N400 component measure?",
         relevant=["en-n400", "hi-n400", "bn-n400"],
         answer="The N400 is a negative ERP around 400 ms that indexes semantic processing and language comprehension."),
    dict(id="q02", lang="en", type="factoid",
         query="How is the P300 used in a brain-computer interface?",
         relevant=["en-p300", "hi-p300"],
         answer="The P300 elicited in the oddball paradigm drives the P300 speller, letting users select letters without movement."),
    dict(id="q03", lang="en", type="factoid",
         query="What frequency is the alpha rhythm and when does it appear?",
         relevant=["en-alpha", "hi-alpha", "ta-alpha"],
         answer="Alpha is an 8 to 12 Hz occipital rhythm that appears with eyes closed and relaxation and drops with attention."),
    dict(id="q04", lang="en", type="factoid",
         query="Why is ICA used in EEG preprocessing?",
         relevant=["en-ica", "hi-ica", "en-preproc"],
         answer="ICA separates EEG into independent sources so that eye-blink and muscle artifacts can be removed."),
    dict(id="q05", lang="en", type="factoid",
         query="Which EEG changes are seen in Alzheimer's disease?",
         relevant=["en-alzheimer", "hi-alzheimer"],
         answer="Alzheimer's shows EEG slowing: more delta and theta, less alpha and beta, plus reduced connectivity."),
    dict(id="q06", lang="en", type="factoid",
         query="How does the mu rhythm support motor-imagery BCIs?",
         relevant=["en-mu", "hi-mu", "ta-bci"],
         answer="The mu rhythm desynchronizes during imagined movement, providing a control signal for motor-imagery BCIs."),
    dict(id="q07", lang="en", type="global",
         query="Which ERP components are altered in schizophrenia?",
         relevant=["en-mmn", "en-p300"],
         answer="Mismatch negativity and P300 are both reduced in schizophrenia and studied as biomarkers."),
    dict(id="q08", lang="en", type="multi-hop",
         query="What signals and methods are used to build a motor-imagery brain-computer interface?",
         relevant=["en-mu", "en-csp", "en-dl", "ta-bci"],
         answer="Motor-imagery BCIs use mu-rhythm desynchronization with features from common spatial patterns or deep networks like EEGNet."),
    dict(id="q09", lang="en", type="global",
         query="How is EEG used to study and stage sleep?",
         relevant=["en-delta", "hi-delta", "en-sleep"],
         answer="Sleep staging uses delta waves to mark deep NREM and spindles and K-complexes to mark stage 2."),
    dict(id="q10", lang="en", type="factoid",
         query="What is time-frequency analysis good for compared with averaging?",
         relevant=["en-tf", "en-erp-method"],
         answer="Time-frequency analysis reveals transient band-specific bursts that trial averaging cancels out."),
    dict(id="q11", lang="en", type="factoid",
         query="What is the theta/beta ratio associated with?",
         relevant=["en-adhd", "hi-adhd"],
         answer="An elevated frontal theta/beta ratio has been reported in ADHD, though its specificity is limited."),
    dict(id="q12", lang="en", type="factoid",
         query="How does source localization work for EEG?",
         relevant=["en-source"],
         answer="Source localization solves an ill-posed inverse problem with a head model to map scalp EEG to cortical generators."),

    # cross-lingual queries (Hindi query, relevant docs include English)
    dict(id="q13", lang="hi", type="cross-lingual",
         query="N400 घटक क्या मापता है?",
         relevant=["hi-n400", "en-n400", "bn-n400"],
         answer="N400 लगभग 400 ms का ऋणात्मक ERP है जो अर्थपूर्ण प्रसंस्करण और भाषा बोध का सूचकांक है।"),
    dict(id="q14", lang="hi", type="cross-lingual",
         query="अल्फा लय की आवृत्ति क्या है और यह कब दिखाई देती है?",
         relevant=["hi-alpha", "en-alpha", "ta-alpha"],
         answer="अल्फा 8 से 12 हर्ट्ज़ की पश्चकपाल लय है जो आँखें बंद करने पर दिखती है और ध्यान के साथ घटती है।"),
    dict(id="q15", lang="hi", type="cross-lingual",
         query="मिर्गी के निदान में EEG कैसे मदद करता है?",
         relevant=["hi-epilepsy", "en-epilepsy", "bn-epilepsy"],
         answer="EEG मिर्गी में अंतराल-काल के स्पाइक और दौरे के दौरान समकालिक विसर्जन दिखाकर निदान में मदद करता है।"),
    dict(id="q16", lang="hi", type="cross-lingual",
         query="मोटर कल्पना BCI में म्यू लय की क्या भूमिका है?",
         relevant=["hi-mu", "en-mu", "ta-bci"],
         answer="मोटर कल्पना के दौरान म्यू लय दब जाती है, जो BCI के लिए नियंत्रण संकेत प्रदान करती है।"),
    dict(id="q17", lang="hi", type="cross-lingual",
         query="अल्ज़ाइमर रोग में EEG में क्या परिवर्तन होते हैं?",
         relevant=["hi-alzheimer", "en-alzheimer"],
         answer="अल्ज़ाइमर में EEG धीमा हो जाता है: डेल्टा और थीटा बढ़ती है, अल्फा और बीटा घटती है।"),

    # Bengali / Tamil cross-lingual
    dict(id="q18", lang="bn", type="cross-lingual",
         query="N400 কী পরিমাপ করে?",
         relevant=["bn-n400", "en-n400", "hi-n400"],
         answer="N400 হল একটি ঋণাত্মক ERP যা অর্থগত প্রক্রিয়াকরণ ও ভাষা বোঝার সূচক।"),
    dict(id="q19", lang="ta", type="cross-lingual",
         query="மியூ ரிதம் மூளை-கணினி இடைமுகத்தில் எவ்வாறு உதவுகிறது?",
         relevant=["ta-bci", "en-mu", "hi-mu"],
         answer="மோட்டார் கற்பனையின்போது மியூ ரிதம் அடக்கப்பட்டு BCI கட்டுப்பாட்டு சமிக்ஞையாகச் செயல்படுகிறது."),

    # more global / multi-hop
    dict(id="q20", lang="en", type="global",
         query="What are the standard steps of an EEG preprocessing pipeline?",
         relevant=["en-preproc", "en-ica", "en-erp-method"],
         answer="Band-pass filter, re-reference, reject bad segments, run ICA for artifacts, then epoch before averaging."),
    dict(id="q21", lang="en", type="factoid",
         query="What is the P600 associated with in language?",
         relevant=["en-p600"],
         answer="The P600 is a late positivity linked to syntactic reanalysis and grammatical violations."),
    dict(id="q22", lang="en", type="factoid",
         query="What generates theta oscillations and what cognition do they support?",
         relevant=["en-theta", "hi-theta"],
         answer="Theta is generated partly by the hippocampus and supports working memory and spatial navigation."),
    dict(id="q23", lang="en", type="global",
         query="How can EEG deep learning models decode brain activity?",
         relevant=["en-dl", "hi-dl", "en-csp"],
         answer="CNN/RNN models like EEGNet learn spatio-temporal filters to decode motor imagery, sleep, and seizures."),
    dict(id="q24", lang="en", type="factoid",
         query="What is frontal alpha asymmetry a marker of?",
         relevant=["en-depression", "hi-depression"],
         answer="Frontal alpha asymmetry is proposed as an EEG marker of depression and approach/withdrawal motivation."),
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    corpus_path = ROOT / "data" / "corpus" / "neuro_corpus.jsonl"
    queries_path = ROOT / "data" / "eval" / "queries.jsonl"

    # sanity: every relevance judgment must reference an existing doc id
    doc_ids = {d["id"] for d in DOCS}
    for q in QUERIES:
        missing = [r for r in q["relevant"] if r not in doc_ids]
        if missing:
            raise ValueError(f"query {q['id']} references unknown docs: {missing}")

    _write_jsonl(corpus_path, DOCS)
    _write_jsonl(queries_path, QUERIES)
    print(f"wrote {len(DOCS)} passages  -> {corpus_path.relative_to(ROOT)}")
    print(f"wrote {len(QUERIES)} queries  -> {queries_path.relative_to(ROOT)}")
    langs = sorted({d['lang'] for d in DOCS})
    print(f"languages: {', '.join(langs)}")


if __name__ == "__main__":
    main()
