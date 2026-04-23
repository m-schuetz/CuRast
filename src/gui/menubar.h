
void CuRast::makeMenubar(){

	if (ImGui::BeginMainMenuBar()){

		ImGui::MenuItem(" ##hidebuttondoesntworkwithoutthis", "");

		ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 65);

		if(ImGui::MenuItem("Hide GUI", "")){
			CuRastSettings::hideGUI = !CuRastSettings::hideGUI;
		}

		ImGui::EndMainMenuBar();
	}

}